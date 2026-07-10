"""Homography solving: template assignment search, scoring, refinement.

Approach
--------
Detected straight lines are clustered into (at most) two pencils by their
vanishing points -- the images of the pitch's two orthogonal line families.
Candidate assignments of detected lines to template lines are enumerated
(order within a pencil is projectively preserved, so only monotone
assignments are considered), solved by mixed point/line DLT, and scored
globally against the observed marking pixels via a distance transform
(chamfer fit + fraction of observed pixels explained).

A dedicated strategy handles the classic broadcast midfield view -- centre
circle + halfway line + one longitudinal line -- where the longitudinal
vanishing point is under-determined: it is recovered by a 1-D search along
the longitudinal line, using the two tangent lines from the candidate
vanishing point to the detected ellipse (which correspond to the template
lines y = cy -/+ r tangent to the centre circle).

Mirror-image solutions are rejected by the handedness test (physically
impossible), and the remaining 180-degree ambiguity is resolved by
canonicalising the camera side (see :mod:`pitchcal.geometry`).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import DetectedFeatures, DetectedLine
from .geometry import (
    apply_h_w,
    dlt_homography,
    dlt_homography_family,
    intersect_lines,
    is_proper_view,
    line_conic_intersections,
    normalize_line,
    tangent_lines_from_point,
    to_h,
)
from .pitch_model import CIRCLE_RADIUS, PitchModel, TemplateLine


# ---------------------------------------------------------------------- #
# Vanishing-point clustering
# ---------------------------------------------------------------------- #
def _vp_residual_deg(line: DetectedLine, vp: np.ndarray) -> float:
    """Angle between a line's direction and the direction to a VP."""
    d = line.direction
    if abs(vp[2]) < 1e-9 * max(abs(vp[0]), abs(vp[1]), 1e-12):
        t = np.array([vp[0], vp[1]])
    else:
        t = np.array([vp[0] / vp[2], vp[1] / vp[2]]) - line.midpoint
    n = np.linalg.norm(t)
    if n < 1e-9:
        return 90.0
    cosang = abs(float(d @ t) / n)
    return float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))


def fit_pencil_vp(
    lines: list[DetectedLine], idxs: list[int], tol_deg: float = 2.5
) -> np.ndarray | None:
    """Best-fit common vanishing point of a set of lines, or None.

    Returns the VP (homogeneous) if every line passes within ``tol_deg``
    of it, i.e. the set is consistent as one pencil.
    """
    if len(idxs) < 2:
        return None
    a, b = sorted(idxs, key=lambda i: -lines[i].length)[:2]
    v = intersect_lines(lines[a].l, lines[b].l)
    if np.linalg.norm(v) < 1e-12:
        return None
    for i in idxs:
        if _vp_residual_deg(lines[i], v) > tol_deg:
            return None
    return v


def pencil_order(lines: list[DetectedLine], idxs: list[int], vp: np.ndarray) -> list[int]:
    """Sort a pencil's line indices in sweep order around the VP."""
    if not idxs:
        return []
    mids = np.array([lines[i].midpoint for i in idxs])
    if abs(vp[2]) < 1e-9 * max(abs(vp[0]), abs(vp[1]), 1e-12):
        # Parallel pencil: order by perpendicular offset along the normal.
        d = np.array([vp[0], vp[1]])
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        keys = mids @ n
    else:
        v = np.array([vp[0] / vp[2], vp[1] / vp[2]])
        ref = mids.mean(axis=0) - v
        ref = ref / max(np.linalg.norm(ref), 1e-9)
        rel = mids - v
        keys = np.arctan2(
            rel[:, 0] * ref[1] - rel[:, 1] * ref[0], rel @ ref
        )
    return [idxs[k] for k in np.argsort(keys)]


# ---------------------------------------------------------------------- #
# Global chamfer scorer
# ---------------------------------------------------------------------- #
class Scorer:
    """Distance-transform scoring of a candidate H against observed pixels."""

    def __init__(
        self,
        features: DetectedFeatures,
        pitch: PitchModel,
        step: float = 0.75,
        max_grid: int = 1600,
    ) -> None:
        self.pitch = pitch
        self.origin = features.canvas_origin
        cw, ch = features.canvas_size
        self.scale = min(1.0, max_grid / max(cw, ch, 1))
        self.gw = max(2, int(cw * self.scale))
        self.gh = max(2, int(ch * self.scale))
        mask = np.zeros((self.gh, self.gw), np.uint8)
        pts = ((features.line_points - self.origin) * self.scale).astype(int)
        ok = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < self.gw) & (pts[:, 1] < self.gh)
        mask[pts[ok, 1], pts[ok, 0]] = 1
        self.dt = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3).astype(np.float32)
        self.cap = 18.0 * self.scale
        # Template curve samples, grouped per curve for polyline drawing.
        self.curves: list[np.ndarray] = [pts_ for _, pts_ in pitch.sample_all(step)]
        self.samples = np.concatenate(self.curves)
        det = features.line_points
        if len(det) > 3000:
            det = det[:: len(det) // 3000]
        self.det_grid = (det - self.origin) * self.scale
        # Frame rectangle (in grid coords) for visibility bookkeeping.
        self.frame_lo = (np.array([0.0, 0.0]) - self.origin) * self.scale
        self.frame_hi = (
            np.array(features.image_size, dtype=float) - self.origin
        ) * self.scale

    def _project(self, H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p, w = apply_h_w(H, pts)
        g = (p - self.origin) * self.scale
        inside = (
            (w > 1e-9)
            & (g[:, 0] >= 0)
            & (g[:, 1] >= 0)
            & (g[:, 0] < self.gw - 1)
            & (g[:, 1] < self.gh - 1)
        )
        return g, inside

    def fit(self, H: np.ndarray, min_inside: int = 60) -> tuple[float, int]:
        """Trimmed mean capped DT distance (in [0, 1]) at template samples.

        The worst quarter of samples is dropped: real frames always have
        template stretches with no detected support (dashed, faint or
        occluded paint), and a correct homography should not be charged
        for detection recall.  A wrong assignment misplaces far more than
        a quarter of the template, so it still scores badly.
        """
        g, inside = self._project(H, self.samples)
        n = int(inside.sum())
        if n < min_inside:
            return float("inf"), n
        gi = g[inside].astype(int)
        d = np.minimum(self.dt[gi[:, 1], gi[:, 0]], self.cap)
        d.sort()
        keep = max(min_inside // 2, int(0.75 * n))
        return float(d[:keep].mean() / self.cap), n

    def explained(self, H: np.ndarray) -> float:
        """Fraction of observed marking pixels near a projected template curve.

        Only pixels that this candidate itself claims lie on (or near) the
        pitch are counted: real frames contain legitimate white markings
        off the pitch (technical-area boxes, warm-up lines, signage) that
        no correct homography should be punished for, while a *wrong*
        assignment still gets punished for the unexplained pitch markings
        it places inside the pitch.
        """
        canvas = np.zeros((self.gh, self.gw), np.uint8)
        for curve in self.curves:
            g, inside = self._project(H, curve)
            gi = np.round(g).astype(int)
            run: list[np.ndarray] = []
            for k in range(len(curve)):
                if inside[k]:
                    run.append(gi[k])
                elif run:
                    if len(run) > 1:
                        cv2.polylines(canvas, [np.array(run)], False, 1, 1)
                    run = []
            if len(run) > 1:
                cv2.polylines(canvas, [np.array(run)], False, 1, 1)
        if canvas.max() == 0:
            return 0.0
        dt2 = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 3)
        gp = self.det_grid
        ok = (
            (gp[:, 0] >= 0)
            & (gp[:, 1] >= 0)
            & (gp[:, 0] < self.gw - 1)
            & (gp[:, 1] < self.gh - 1)
        )
        # Restrict to pixels that map onto the pitch under this candidate.
        det_pts = gp / self.scale + self.origin
        pp, wq = apply_h_w(np.linalg.inv(H), det_pts)
        on_pitch = (
            (np.abs(wq) > 1e-12)
            & (pp[:, 0] > -1.5)
            & (pp[:, 0] < self.pitch.length + 1.5)
            & (pp[:, 1] > -1.5)
            & (pp[:, 1] < self.pitch.width + 1.5)
        )
        n_in_canvas = int(ok.sum())
        ok &= on_pitch
        if ok.sum() < 20 or n_in_canvas < 20:
            return 0.0
        # A plausible broadcast homography places most detected marking
        # pixels on the pitch; a mapping that pushes nearly everything
        # off-pitch is explaining nothing however well its sliver aligns.
        coverage = float(ok.sum()) / float(n_in_canvas)
        gi = gp[ok].astype(int)
        frac_near = float((dt2[gi[:, 1], gi[:, 0]] < max(4.0, 8.0 * self.scale)).mean())
        return frac_near * min(1.0, coverage / 0.30)

    def worst_line_explained(
        self,
        H: np.ndarray,
        lines: list[DetectedLine],
        ellipse_points: np.ndarray | None = None,
        min_support_frac: float = 0.15,
    ) -> float:
        """Explained fraction of the worst *major* detected structure.

        A correct homography explains every substantial detected line (and
        the detected circle) that lies on the pitch; a wrong assignment
        strands at least one whole structure.  Returns 1.0 when no major
        on-pitch structure exists.
        """
        canvas = np.zeros((self.gh, self.gw), np.uint8)
        for curve in self.curves:
            g, inside = self._project(H, curve)
            gi = np.round(g).astype(int)
            run: list[np.ndarray] = []
            for k in range(len(curve)):
                if inside[k]:
                    run.append(gi[k])
                elif run:
                    if len(run) > 1:
                        cv2.polylines(canvas, [np.array(run)], False, 1, 1)
                    run = []
            if len(run) > 1:
                cv2.polylines(canvas, [np.array(run)], False, 1, 1)
        if canvas.max() == 0:
            return 0.0
        dt2 = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 3)
        total = sum(ln.support for ln in lines)
        if total == 0:
            return 1.0
        Hinv = np.linalg.inv(H)
        worst = 1.0
        thr = max(4.0, 8.0 * self.scale)
        structures: list[np.ndarray] = [
            ln.points for ln in lines if ln.support >= min_support_frac * total
        ]
        if ellipse_points is not None and len(ellipse_points) >= 100:
            structures.append(ellipse_points)
        for spts in structures:
            pts = spts[:: max(1, len(spts) // 300)]
            pp, wq = apply_h_w(Hinv, pts)
            on = (
                (np.abs(wq) > 1e-12)
                & (pp[:, 0] > -1.5)
                & (pp[:, 0] < self.pitch.length + 1.5)
                & (pp[:, 1] > -1.5)
                & (pp[:, 1] < self.pitch.width + 1.5)
            )
            if on.mean() < 0.5:
                continue  # mostly off-pitch (e.g. ad-board edge): not vetoed
            g = (pts[on] - self.origin) * self.scale
            okg = (
                (g[:, 0] >= 0)
                & (g[:, 1] >= 0)
                & (g[:, 0] < self.gw - 1)
                & (g[:, 1] < self.gh - 1)
            )
            if okg.sum() < 15:
                continue
            gi = g[okg].astype(int)
            frac = float((dt2[gi[:, 1], gi[:, 0]] < thr).mean())
            worst = min(worst, frac)
        return worst

    def dt_bilinear(self, g: np.ndarray) -> np.ndarray:
        x0 = np.floor(g[:, 0]).astype(int)
        y0 = np.floor(g[:, 1]).astype(int)
        fx = g[:, 0] - x0
        fy = g[:, 1] - y0
        d = (
            self.dt[y0, x0] * (1 - fx) * (1 - fy)
            + self.dt[y0, x0 + 1] * fx * (1 - fy)
            + self.dt[y0 + 1, x0] * (1 - fx) * fy
            + self.dt[y0 + 1, x0 + 1] * fx * fy
        )
        return d


# ---------------------------------------------------------------------- #
# Candidate representation
# ---------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One hypothesised pitch->image homography with its evidence."""

    H: np.ndarray
    strategy: str
    assignment: dict[int, str]
    dlt_residual: float
    n_constraints: int
    fit: float = float("inf")
    n_inside: int = 0
    explained: float = 0.0
    used_players: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.fit + 0.6 * (1.0 - self.explained)


def _family_constraints(k: int) -> int:
    if k <= 0:
        return 0
    return 2 if k == 1 else k + 2


def view_plausible(
    H: np.ndarray, image_size: tuple[int, int], pitch: PitchModel
) -> bool:
    """Reject mappings whose frame covers an implausible pitch region.

    The visible frame of a broadcast camera maps to a region comparable to
    (a part of) the pitch: not vastly larger than the pitch itself, not a
    few square metres, and not centred far outside the ground.
    """
    w, h = image_size
    Hinv = np.linalg.inv(H)
    quad = np.array(
        [[0.05 * w, 0.3 * h], [0.95 * w, 0.3 * h], [0.95 * w, 0.95 * h], [0.05 * w, 0.95 * h]]
    )
    pp, wq = apply_h_w(Hinv, quad)
    if np.any(np.abs(wq) < 1e-12) or np.any(wq * wq[0] <= 0):
        return False
    x, y = pp[:, 0], pp[:, 1]
    area = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )
    if not (25.0 < area < 1.6 * pitch.length * pitch.width):
        return False
    cx, cy = pp.mean(axis=0)
    return -25.0 < cx < pitch.length + 25.0 and -20.0 < cy < pitch.width + 20.0


def _extent_ok(
    Hinv: np.ndarray, line: DetectedLine, tline: TemplateLine, pitch: PitchModel
) -> bool:
    """Detected support extent must land on the template line's painted part."""
    p, w = apply_h_w(Hinv, np.stack([line.p0, line.p1]))
    if np.any(w * w[0] <= 0):
        return False
    free = p[:, 0] if tline.family == "longitudinal" else p[:, 1]
    a, b = float(min(free)), float(max(free))
    axis_len = pitch.length if tline.family == "longitudinal" else pitch.width
    if a < -25.0 or b > axis_len + 25.0 or b - a < 0.5:
        return False
    slack = 7.0
    overlap = 0.0
    for lo, hi in tline.segments:
        overlap += max(0.0, min(b, hi + slack) - max(a, lo - slack))
    return overlap >= 0.5 * (b - a)


# ---------------------------------------------------------------------- #
# Strategy A: line-assignment enumeration
# ---------------------------------------------------------------------- #
def _circle_point_pairs(
    ellipse_conic: np.ndarray, halfway_img_line: np.ndarray, pitch: PitchModel
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Point pairs from ellipse x halfway-line, in both template orders."""
    ipts = line_conic_intersections(halfway_img_line, ellipse_conic)
    if ipts is None:
        return []
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    pa = np.array([cx, cy - CIRCLE_RADIUS])
    pb = np.array([cx, cy + CIRCLE_RADIUS])
    return [
        [(pa, ipts[0]), (pb, ipts[1])],
        [(pa, ipts[1]), (pb, ipts[0])],
    ]


def enumerate_line_candidates(
    features: DetectedFeatures,
    pitch: PitchModel,
    scorer: Scorer,
    max_hyp: int = 40000,
    max_lines_per_family: int = 5,
    time_budget_s: float = 8.0,
) -> list[Candidate]:
    """Strategy A: enumerate family colorings + monotone assignments, DLT-solve.

    With few detected lines the family structure cannot be clustered
    reliably (any two lines meet somewhere), so all 2-colorings of the
    detected lines into (family-A, family-B) are considered, pruned by
    pencil consistency: >= 3 lines of one color must share a vanishing
    point.  Within a coloring, assignments to template positions are
    monotone in pencil sweep order.
    """
    lines = features.lines
    keep = sorted(range(len(lines)), key=lambda i: -lines[i].length)[: 2 * max_lines_per_family]
    if not keep:
        return []
    t_start = time.time()

    ell = features.ellipse
    conic = ell.conic if ell is not None else None

    long_names = [l.name for l in pitch.family_lines("longitudinal")]
    trans_names = [l.name for l in pitch.family_lines("transverse")]

    # Conic point sources: template lines whose intersection with a known
    # circle/arc yields identified point correspondences.  Each entry:
    # (required template line, the two template points sorted by the free
    # coordinate, elation circle name, elation fixed x).
    arc_dy = float(np.sqrt(CIRCLE_RADIUS**2 - 5.5**2))  # pen-front chord half-length
    cx_, cy_ = pitch.length / 2.0, pitch.width / 2.0
    conic_sources: list[tuple[str, np.ndarray, np.ndarray, str, float]] = []
    if ell is not None:
        if ell.kind == "circle":
            conic_sources.append(
                (
                    "halfway",
                    np.array([cx_, cy_ - CIRCLE_RADIUS]),
                    np.array([cx_, cy_ + CIRCLE_RADIUS]),
                    "center_circle",
                    cx_,
                )
            )
        else:  # a partial arc: could be either penalty arc or a cut circle
            conic_sources.extend(
                [
                    (
                        "pen_front_left",
                        np.array([16.5, cy_ - arc_dy]),
                        np.array([16.5, cy_ + arc_dy]),
                        "pen_arc_left",
                        16.5,
                    ),
                    (
                        "pen_front_right",
                        np.array([pitch.length - 16.5, cy_ - arc_dy]),
                        np.array([pitch.length - 16.5, cy_ + arc_dy]),
                        "pen_arc_right",
                        pitch.length - 16.5,
                    ),
                    (
                        "halfway",
                        np.array([cx_, cy_ - CIRCLE_RADIUS]),
                        np.array([cx_, cy_ + CIRCLE_RADIUS]),
                        "center_circle",
                        cx_,
                    ),
                ]
            )

    colorings: list[tuple[list[int], list[int]]] = []
    n = len(keep)
    for mask in range(2 ** max(n - 1, 0)):  # line keep[0] fixed to color A
        fam_a = [keep[0]] + [keep[i] for i in range(1, n) if not (mask >> (i - 1)) & 1]
        fam_b = [keep[i] for i in range(1, n) if (mask >> (i - 1)) & 1]
        fam_a = fam_a[:max_lines_per_family]
        fam_b = fam_b[:max_lines_per_family]
        ok = True
        ordered: list[list[int]] = []
        for fam in (fam_a, fam_b):
            if len(fam) >= 2:
                vp = fit_pencil_vp(lines, fam)
                if vp is None:
                    ok = False
                    break
                ordered.append(pencil_order(lines, fam, vp))
            else:
                ordered.append(fam)
        if ok:
            colorings.append((ordered[0], ordered[1]))

    def assignments_by_k(det: list[int], names: list[str]) -> dict[int, list[dict[int, str]]]:
        """Monotone det->template assignments grouped by size k.

        ``det`` is in pencil sweep order and ``names`` in template-position
        order; projective maps preserve order up to reversal, so only
        order-preserving and order-reversing assignments are generated.
        """
        by_k: dict[int, list[dict[int, str]]] = {0: [{}]}
        for k in range(1, min(len(det), len(names)) + 1):
            lst: list[dict[int, str]] = []
            for dsub in itertools.combinations(det, k):
                for tsub in itertools.combinations(names, k):
                    lst.append(dict(zip(dsub, tsub)))
                    if k >= 2:
                        lst.append(dict(zip(dsub, tsub[::-1])))
            by_k[k] = lst
        return by_k

    # Generate specs richest-first under a budget; the conic (if any) can
    # add up to 4 constraints when the halfway line is in the assignment.
    specs: list[tuple[int, dict[int, str]]] = []
    seen: set[tuple] = set()
    conic_bonus = 4 if conic is not None else 0
    for fam_a, fam_b in colorings:
        for a_names, b_names in ((trans_names, long_names), (long_names, trans_names)):
            by_a = assignments_by_k(fam_a, a_names)
            by_b = assignments_by_k(fam_b, b_names)
            pairs = [
                (ka, kb)
                for ka in by_a
                for kb in by_b
                if ka + kb >= 2
                and _family_constraints(ka) + _family_constraints(kb) + conic_bonus >= 8
            ]
            pairs.sort(key=lambda t: -(t[0] + t[1]))
            for ka, kb in pairs:
                for asg_a in by_a[ka]:
                    for asg_b in by_b[kb]:
                        key = tuple(sorted({**asg_a, **asg_b}.items()))
                        if key in seen:
                            continue
                        seen.add(key)
                        specs.append((ka + kb, {**asg_a, **asg_b}))
                if len(specs) >= max_hyp:
                    break
            if len(specs) >= max_hyp:
                break
        if len(specs) >= max_hyp:
            break
    uniq = sorted(specs, key=lambda t: -t[0])[:max_hyp]

    name_to_line = {l.name: l for l in pitch.lines}
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    candidates: list[Candidate] = []
    best_fit = float("inf")
    for k, asg in uniq:
        line_pairs = []
        for det_i, tname in asg.items():
            tl = name_to_line[tname]
            pa, pb = tl.endpoints(pitch)
            line_pairs.append((lines[det_i].l, (pa, pb)))
        # Conic-derived point options: [] (no conic use) plus, for each
        # applicable source, the two intersection orderings.
        point_sets: list[tuple[list[tuple[np.ndarray, np.ndarray]], tuple | None]] = [([], None)]
        n_transverse = sum(
            1 for nm in asg.values() if name_to_line[nm].family == "transverse"
        )
        for req_name, Pa, Pb, circ_name, fix_x in conic_sources:
            det_i = next((i for i, n in asg.items() if n == req_name), None)
            if det_i is None or conic is None:
                continue
            ipts = line_conic_intersections(lines[det_i].l, conic)
            if ipts is None:
                continue
            elation = (circ_name, fix_x) if n_transverse < 3 else None
            point_sets.append(([(Pa, ipts[0]), (Pb, ipts[1])], elation))
            point_sets.append(([(Pa, ipts[1]), (Pb, ipts[0])], elation))
        for pts_pairs, elation in point_sets:
            n_con = (
                sum(_family_constraints(x) for x in _asg_family_sizes(asg, name_to_line))
                + 2 * len(pts_pairs)
            )
            # >= 7: one-parameter families are sampled and resolved by the
            # global chamfer score (or the elation correction).
            if n_con < 7:
                continue
            def _fam_score(Hc: np.ndarray) -> float:
                if not is_proper_view(
                    Hc, pitch.length, pitch.width, features.image_size
                ):
                    return float("inf")
                return scorer.fit(Hc)[0]

            # Score-guided family refinement is expensive; only invest in it
            # while no good candidate exists yet.
            solutions = dlt_homography_family(
                point_pairs=list(pts_pairs),
                line_pairs=line_pairs,
                score_fn=_fam_score if best_fit > 0.15 else None,
            )
            if len(solutions) > 8:
                solutions = solutions[::4]
            best_member: Candidate | None = None
            for H, resid in solutions:
                # Assignments whose only transverse evidence is the conic's
                # own line leave an x-scaling elation open even with the
                # intersection points (those lie ON that line); the conic's
                # x-extent closes it.  Harmless when already pinned.
                if elation is not None:
                    ec = _elation_correction(
                        H, conic, pitch, fix_x=elation[1], circle=elation[0]
                    )
                    if ec is not None:
                        H = ec[0]
                if not is_proper_view(H, pitch.length, pitch.width, features.image_size):
                    continue
                if not view_plausible(H, features.image_size, pitch):
                    continue
                Hinv = np.linalg.inv(H)
                if not all(
                    _extent_ok(Hinv, lines[i], name_to_line[nm], pitch)
                    for i, nm in asg.items()
                ):
                    continue
                fit, n_in = scorer.fit(H)
                if not np.isfinite(fit):
                    continue
                cand = Candidate(
                    H=H,
                    strategy="lines",
                    assignment=dict(asg),
                    dlt_residual=resid,
                    n_constraints=n_con,
                    fit=fit,
                    n_inside=n_in,
                )
                if best_member is None or cand.fit < best_member.fit:
                    best_member = cand
            if best_member is not None:
                candidates.append(best_member)
                best_fit = min(best_fit, best_member.fit)
        if best_fit < 0.05 and k >= 5 and len(candidates) > 30:
            break
        if time.time() - t_start > time_budget_s:
            break
    return candidates


def _asg_family_sizes(asg: dict[int, str], name_to_line: dict[str, TemplateLine]) -> list[int]:
    ka = sum(1 for n in asg.values() if name_to_line[n].family == "transverse")
    kb = sum(1 for n in asg.values() if name_to_line[n].family == "longitudinal")
    return [ka, kb]


# ---------------------------------------------------------------------- #
# Strategy B: circle + halfway + one longitudinal line
# ---------------------------------------------------------------------- #
def _elation_correction(
    H: np.ndarray,
    conic_img: np.ndarray,
    pitch: PitchModel,
    fix_x: float | None = None,
    circle: str = "center_circle",
) -> tuple[np.ndarray, float] | None:
    """Fix the x-scaling elation left open by conic-on-a-line constraints.

    When an assignment's only transverse evidence is one line (halfway, or
    a penalty-front) plus conic intersection points *on that line*, every
    correspondence is invariant under the elation x' = a(x - fix_x) +
    fix_x, so DLT leaves a one-parameter family.  The detected conic's
    x-extent is the evidence that pins it: pull the image conic back to
    pitch space, read its two x-tangent lines, and solve the scaling that
    maps them onto the template circle's tangents x = cx0 -/+ r.

    Args:
        H: candidate pitch->image homography.
        conic_img: detected ellipse conic (image coords).
        pitch: template.
        fix_x: x of the pointwise-fixed transverse line (default: pitch
            centre, the halfway line).
        circle: template circle the conic corresponds to ("center_circle",
            "pen_arc_left" or "pen_arc_right").

    Returns (H_corrected, circle_rms) with circle_rms the conic-space
    mismatch vs the template circle (0 = perfect), or None if the
    pull-back is not sane.
    """
    arc = next(a for a in pitch.arcs if a.name == circle)
    cx0, _cy0 = arc.center
    r = arc.radius
    if fix_x is None:
        fix_x = pitch.length / 2.0
    Q = H.T @ conic_img @ H  # pull-back of the image conic to pitch coords
    Q = Q / np.linalg.norm(Q)
    try:
        Qd = np.linalg.inv(Q)  # dual conic (scale-free)
    except np.linalg.LinAlgError:
        return None
    # Tangent lines of the form (1, 0, -x0):  Qd00 - 2 x0 Qd02 + x0^2 Qd22 = 0.
    A2, B2, C2 = Qd[2, 2], -2.0 * Qd[0, 2], Qd[0, 0]
    disc = B2 * B2 - 4.0 * A2 * C2
    if abs(A2) < 1e-15 or disc <= 0:
        return None
    x_meas = np.array(
        [(-B2 - np.sqrt(disc)) / (2.0 * A2), (-B2 + np.sqrt(disc)) / (2.0 * A2)]
    )
    if not np.isfinite(x_meas).all() or abs(x_meas[1] - x_meas[0]) < 1.0:
        return None
    x_tgt = np.array(sorted([cx0 - r, cx0 + r]))
    x_meas = np.sort(x_meas)
    # Least-squares scaling about fix_x mapping measured -> target tangents.
    u = x_meas - fix_x
    v = x_tgt - fix_x
    denom = float(u @ u)
    if denom < 1e-9:
        return None
    a = float(u @ v) / denom
    if not (0.05 < a < 20.0):
        return None
    # H maps pitch->image; pre-compose with the elation that *undoes* the
    # measured-vs-target discrepancy: E maps target positions to measured.
    if abs(a) < 1e-9:
        return None
    E = np.array(
        [[1.0 / a, 0.0, fix_x * (1.0 - 1.0 / a)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    Hc = H @ E
    Hc = Hc / np.linalg.norm(Hc)
    # Residual: corrected pull-back vs the exact template circle.
    Qc = Hc.T @ conic_img @ Hc
    Qc = Qc / np.linalg.norm(Qc)
    K = pitch.circle_conic(circle)
    K = K / np.linalg.norm(K)
    rms = float(min(np.linalg.norm(Qc - K), np.linalg.norm(Qc + K)))
    return Hc, rms
def _projective_samples_on_line(
    line: DetectedLine, diag: float, n: int = 96
) -> list[np.ndarray]:
    """Homogeneous sample points along a line, covering +/- infinity."""
    m = line.midpoint
    d = line.direction
    out: list[np.ndarray] = []
    for u in np.linspace(-0.98, 0.98, n):
        t = np.tan(u * np.pi / 2.0) * 2.0 * diag
        p = m + t * d
        out.append(np.array([p[0], p[1], 1.0]))
    out.append(np.array([d[0], d[1], 0.0]))  # point at infinity
    return out


def _halfway_candidates(features: DetectedFeatures) -> list[int]:
    """Detected lines that pass near the ellipse centre and cut the ellipse."""
    ell = features.ellipse
    if ell is None:
        return []
    return [
        i
        for i, ln in enumerate(features.lines)
        if abs(float(to_h(ell.center[None])[0] @ ln.l)) < 0.5 * ell.axes[1]
        and line_conic_intersections(ln.l, ell.conic) is not None
    ]


def enumerate_circle_candidates(
    features: DetectedFeatures,
    pitch: PitchModel,
    scorer: Scorer,
    keep_top: int = 12,
) -> list[Candidate]:
    """Strategy B: centre circle + halfway + one longitudinal anchor line.

    Four circle points determine the mapping: the ellipse x halfway
    intersections correspond to (cx, cy -/+ r), and the ellipse's extreme
    points along the halfway direction correspond to (cx -/+ r, cy) -- the
    contact points of the transverse tangents.  The first solve
    approximates those extremes with the ellipse-centre chord parallel to
    the halfway line, then a tangent-pin iteration replaces them with the
    true contacts of tangents from the recovered transverse VP.

    Note the anchor line is genuinely needed: without a longitudinal line
    the four (near-)concyclic points plus halfway leave the solution badly
    conditioned, and the frame should fail as under-determined instead.
    """
    ell = features.ellipse
    if ell is None:
        return []
    lines = features.lines
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    r = CIRCLE_RADIUS
    half_idx = _halfway_candidates(features)
    long_names = [l.name for l in pitch.family_lines("longitudinal")]
    name_to_line = {l.name: l for l in pitch.lines}

    out: list[Candidate] = []
    for hi in half_idx:
        point_sets = _circle_point_pairs(ell.conic, lines[hi].l, pitch)
        if not point_sets:
            continue
        half_pa, half_pb = name_to_line["halfway"].endpoints(pitch)
        # Initial guess for the images of (cx -/+ r, cy): ellipse boundary
        # along the halfway direction through the ellipse centre.
        d = lines[hi].direction
        chord = np.cross(
            np.array([ell.center[0], ell.center[1], 1.0]),
            np.array([ell.center[0] + d[0], ell.center[1] + d[1], 1.0]),
        )
        side_pts = line_conic_intersections(chord, ell.conic)
        if side_pts is None:
            continue
        for ai, anchor in enumerate(lines):
            if ai == hi:
                continue
            if line_conic_intersections(anchor.l, ell.conic) is not None:
                continue
            for y0_name in long_names:
                tl = name_to_line[y0_name]
                a_pa, a_pb = tl.endpoints(pitch)
                line_pairs = [
                    (anchor.l, (a_pa, a_pb)),
                    (lines[hi].l, (half_pa, half_pb)),
                ]
                for pts_pairs in point_sets:
                    for s_order in (0, 1):
                        pin_pairs = [
                            (np.array([cx - r, cy]), side_pts[s_order]),
                            (np.array([cx + r, cy]), side_pts[1 - s_order]),
                        ]
                        H, resid = dlt_homography(
                            point_pairs=pts_pairs + pin_pairs, line_pairs=line_pairs
                        )
                        if H is None:
                            continue
                        # Tangent-pin iteration: replace the chord guess by
                        # the true tangency contacts from the recovered
                        # transverse VP, then re-solve.
                        for _ in range(2):
                            vp_t = H @ np.array([0.0, 1.0, 0.0])
                            tg = tangent_lines_from_point(vp_t, ell.conic)
                            if tg is None:
                                break
                            _tl, contacts = tg
                            # Match contacts to pins by proximity.
                            if np.linalg.norm(
                                contacts[0] - pin_pairs[0][1]
                            ) > np.linalg.norm(contacts[1] - pin_pairs[0][1]):
                                contacts = contacts[::-1]
                            pin_pairs = [
                                (np.array([cx - r, cy]), contacts[0]),
                                (np.array([cx + r, cy]), contacts[1]),
                            ]
                            H2, resid = dlt_homography(
                                point_pairs=pts_pairs + pin_pairs,
                                line_pairs=line_pairs,
                            )
                            if H2 is None:
                                break
                            H = H2
                        ec = _elation_correction(H, ell.conic, pitch)
                        circ_rms = float("nan")
                        if ec is not None:
                            H, circ_rms = ec
                        if not is_proper_view(H, pitch.length, pitch.width, features.image_size):
                            continue
                        if not view_plausible(H, features.image_size, pitch):
                            continue
                        Hinv = np.linalg.inv(H)
                        if not _extent_ok(Hinv, anchor, tl, pitch):
                            continue
                        if not _extent_ok(Hinv, lines[hi], name_to_line["halfway"], pitch):
                            continue
                        fit, n_in = scorer.fit(H)
                        if not np.isfinite(fit):
                            continue
                        out.append(
                            Candidate(
                                H=H,
                                strategy="circle",
                                assignment={ai: y0_name, hi: "halfway"},
                                dlt_residual=resid,
                                n_constraints=9,
                                fit=fit,
                                n_inside=n_in,
                                meta={"circle_rms": circ_rms},
                            )
                        )
    out.sort(key=lambda c: c.fit)
    return out[:keep_top]


# ---------------------------------------------------------------------- #
# Strategy C: circle + halfway + player-formation vanishing point
# ---------------------------------------------------------------------- #
def enumerate_formation_candidates(
    features: DetectedFeatures,
    pitch: PitchModel,
    scorer: Scorer,
    formation_lines: list[np.ndarray],
    keep_top: int = 8,
) -> list[Candidate]:
    """Strategy C: use a player defensive-line as a transverse-parallel line.

    A settled defensive line runs (approximately) parallel to the goal
    lines, so its image line passes through the transverse vanishing point.
    Intersecting it with the halfway line gives that VP; the tangents from
    the VP to the centre-circle ellipse correspond to x = Lx/2 -/+ r.  This
    is a *probabilistic* constraint (broken/diagonal defensive lines
    violate it) -- candidates are flagged ``used_players`` and the caller
    reduces confidence accordingly.
    """
    ell = features.ellipse
    if ell is None or ell.kind != "circle" or not formation_lines:
        return []
    lines = features.lines
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    half_idx = [
        i
        for i, ln in enumerate(lines)
        if abs(float(to_h(ell.center[None])[0] @ ln.l)) < 0.5 * ell.axes[1]
        and line_conic_intersections(ln.l, ell.conic) is not None
    ]
    name_to_line = {l.name: l for l in pitch.lines}
    x_tan_a, x_tan_b = cx - CIRCLE_RADIUS, cx + CIRCLE_RADIUS

    out: list[Candidate] = []
    for hi in half_idx:
        point_sets = _circle_point_pairs(ell.conic, lines[hi].l, pitch)
        if not point_sets:
            continue
        half_pa, half_pb = name_to_line["halfway"].endpoints(pitch)
        for fl in formation_lines:
            vp = intersect_lines(normalize_line(fl), lines[hi].l)
            tg = tangent_lines_from_point(vp, ell.conic)
            if tg is None:
                continue
            tangents, contacts = tg
            for t_order in (0, 1):
                t1, t2 = tangents[t_order], tangents[1 - t_order]
                contact_pairs = [
                    (np.array([x_tan_a, cy]), contacts[t_order]),
                    (np.array([x_tan_b, cy]), contacts[1 - t_order]),
                ]
                for pts_pairs in point_sets:
                    line_pairs = [
                        (lines[hi].l, (half_pa, half_pb)),
                        (t1, (np.array([x_tan_a, 0.0]), np.array([x_tan_a, pitch.width]))),
                        (t2, (np.array([x_tan_b, 0.0]), np.array([x_tan_b, pitch.width]))),
                    ]
                    H, resid = dlt_homography(
                        point_pairs=pts_pairs + contact_pairs, line_pairs=line_pairs
                    )
                    if H is None or not is_proper_view(H, pitch.length, pitch.width, features.image_size):
                        continue
                    if not view_plausible(H, features.image_size, pitch):
                        continue
                    if not _extent_ok(
                        np.linalg.inv(H), lines[hi], name_to_line["halfway"], pitch
                    ):
                        continue
                    fit, n_in = scorer.fit(H)
                    if not np.isfinite(fit):
                        continue
                    out.append(
                        Candidate(
                            H=H,
                            strategy="formation",
                            assignment={hi: "halfway"},
                            dlt_residual=resid,
                            n_constraints=9,
                            fit=fit,
                            n_inside=n_in,
                            used_players=True,
                        )
                    )
    out.sort(key=lambda c: c.fit)
    return out[:keep_top]


# ---------------------------------------------------------------------- #
# Refinement
# ---------------------------------------------------------------------- #
def refine_candidate(cand: Candidate, scorer: Scorer, features: DetectedFeatures) -> Candidate:
    """Robust least-squares polish of H against the marking distance field.

    Runs coarse-to-fine: a generous residual cap first (so parts of the
    template stranded tens of pixels from their markings still produce a
    gradient), then a tight pass for precision.
    """
    from scipy.optimize import least_squares

    samples = scorer.samples

    def make_residuals(cap: float):
        def residuals(x: np.ndarray) -> np.ndarray:
            H = x.reshape(3, 3)
            g, inside = scorer._project(H, samples)
            r = np.full(len(samples), cap, dtype=float)
            if inside.any():
                r[inside] = np.minimum(scorer.dt_bilinear(g[inside]), cap)
            gauge = (np.linalg.norm(x) - 1.0) * 100.0
            return np.append(r, gauge)

        return residuals

    x0 = cand.H.flatten() / np.linalg.norm(cand.H)
    try:
        for cap, f_scale in ((3.0 * scorer.cap, 12.0), (scorer.cap, 3.0)):
            sol = least_squares(
                make_residuals(cap),
                x0,
                method="trf",
                loss="soft_l1",
                f_scale=f_scale,
                max_nfev=120,
                x_scale="jac",
            )
            x0 = sol.x / np.linalg.norm(sol.x)
    except Exception:
        return cand
    H = x0.reshape(3, 3)
    H = H / np.linalg.norm(H)
    if not is_proper_view(H, scorer.pitch.length, scorer.pitch.width, features.image_size):
        return cand
    fit, n_in = scorer.fit(H)
    if not np.isfinite(fit) or fit > cand.fit:
        return cand
    out = Candidate(
        H=H,
        strategy=cand.strategy,
        assignment=cand.assignment,
        dlt_residual=cand.dlt_residual,
        n_constraints=cand.n_constraints,
        fit=fit,
        n_inside=n_in,
        explained=cand.explained,
        used_players=cand.used_players,
        meta=cand.meta,
    )
    return out


# ---------------------------------------------------------------------- #
# Candidate comparison (for ambiguity detection)
# ---------------------------------------------------------------------- #
def mapping_discrepancy_m(
    H1: np.ndarray, H2: np.ndarray, image_size: tuple[int, int]
) -> float:
    """Median disagreement (metres) of two image->pitch mappings over the frame."""
    w, h = image_size
    xs, ys = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 6), np.linspace(0.2 * h, 0.95 * h, 5))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    p1, w1 = apply_h_w(np.linalg.inv(H1), pts)
    p2, w2 = apply_h_w(np.linalg.inv(H2), pts)
    ok = (np.abs(w1) > 1e-9) & (np.abs(w2) > 1e-9)
    if ok.sum() < 4:
        return float("inf")
    return float(np.median(np.linalg.norm(p1[ok] - p2[ok], axis=1)))
