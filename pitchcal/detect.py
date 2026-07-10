"""Feature detection: grass mask, white-marking pixels, skeleton chains,
lens-distortion self-calibration, straight lines and the circle/arc ellipse.

Everything downstream of distortion estimation works in *undistorted* pixel
coordinates.  The detector is deliberately model-free (classical CV): it
uses no GPU and its failure modes are transparent.  It is also replaceable:
the solver only consumes :class:`DetectedFeatures`, so a learned line /
keypoint detector can be dropped in by constructing that dataclass from its
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .geometry import DivisionModel, normalize_line


@dataclass
class DetectedLine:
    """A merged straight line in undistorted pixel coordinates."""

    l: np.ndarray  # homogeneous line, (a, b) unit-normalised
    p0: np.ndarray  # support extent endpoint (2,)
    p1: np.ndarray  # support extent endpoint (2,)
    support: int  # number of skeleton pixels backing the line
    points: np.ndarray  # the supporting points (N, 2)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def midpoint(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2.0

    @property
    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([1.0, 0.0])


@dataclass
class DetectedEllipse:
    """A fitted conic in undistorted pixel coordinates."""

    conic: np.ndarray  # 3x3 symmetric, normalised
    center: np.ndarray  # (2,)
    axes: tuple[float, float]  # semi-major, semi-minor
    angle_deg: float
    span_deg: float  # angular coverage of supporting pixels
    support: int
    kind: str  # "circle" (likely centre circle) or "arc" (partial / penalty arc)
    points: np.ndarray  # supporting points (N, 2)


@dataclass
class DetectedFeatures:
    """Everything the solver needs, in undistorted pixel coordinates."""

    lines: list[DetectedLine]
    ellipse: DetectedEllipse | None
    lam: float
    dm: DivisionModel
    line_points: np.ndarray  # all marking skeleton pixels (M, 2), undistorted
    image_size: tuple[int, int]  # (w, h) of the original image
    canvas_origin: np.ndarray  # top-left of the undistorted working canvas
    canvas_size: tuple[int, int]  # (w, h) of the working canvas
    field_fraction: float  # fraction of the frame covered by the grass mask
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Grass / field mask
# ---------------------------------------------------------------------- #
def field_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the playing surface (largest green region, holes filled)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h > 30) & (h < 95) & (s > 55) & (v > 45)
    if green.mean() < 0.02:
        return np.zeros(bgr.shape[:2], np.uint8)
    # Refine around the dominant grass hue; saturation/value floors are
    # kept moderate so dusk / shadowed parts of the pitch stay included
    # (speckle-open + largest-component below reject crowd noise).
    hue_peak = int(np.bincount(h[green].ravel(), minlength=180).argmax())
    mask = (
        (np.abs(h.astype(int) - hue_peak) < 14) & (s > 40) & (v > 32)
    ).astype(np.uint8) * 255
    # Kill isolated speckle (crowd pixels) before any closing can merge it.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    # Bridge the painted lines (excluded by the saturation test) so the
    # grass is one component, then keep the largest.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros(bgr.shape[:2], np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    # Fill interior holes (players, markings) via flood fill from the border.
    ff = mask.copy()
    hh, ww = mask.shape
    ffmask = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


# ---------------------------------------------------------------------- #
# White-marking pixels
# ---------------------------------------------------------------------- #
def marking_mask(bgr: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Thin, bright, low-saturation structures inside the field mask."""
    # min(R, B) suppresses grass (low R and B) and mowing-stripe boundaries
    # (which differ mostly in G), while white paint stays bright.
    gray = np.minimum(bgr[..., 2], bgr[..., 0])
    tophat = np.zeros_like(gray)
    for k in (9, 21):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = np.maximum(tophat, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    inner = cv2.erode(field, np.ones((7, 7), np.uint8))
    cand = (tophat > 22) & (hsv[..., 1] < 120) & (bgr[..., 1] > 80) & (inner > 0)
    if cand.mean() < 0.0008:  # weak contrast frame: relax
        cand = (tophat > 13) & (hsv[..., 1] < 140) & (bgr[..., 1] > 60) & (inner > 0)
    mask = cand.astype(np.uint8) * 255

    # Remove blobby components (players, glare): markings are thin, so the
    # max in-component distance-transform value must be small; small compact
    # blobs must be elongated to survive.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    keep = np.zeros(n, bool)
    diag = float(np.hypot(*mask.shape))
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 25:
            continue
        comp = lab == i
        width = 2.0 * float(dt[comp].max())
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        extent = float(np.hypot(bw, bh))
        if width > max(14.0, 0.014 * diag):
            continue
        if extent < 40 and extent < 4.0 * width:
            continue
        keep[i] = True
    return (keep[lab]).astype(np.uint8) * 255


# ---------------------------------------------------------------------- #
# Skeleton chains
# ---------------------------------------------------------------------- #
def skeleton_chains(mask: np.ndarray, min_px: int = 25) -> list[np.ndarray]:
    """Connected chains of marking-skeleton pixels, each (N, 2) as (x, y).

    Compact squiggle chains are dropped: a skeletonised player outline is a
    dense closed loop whose pixel count far exceeds its bounding-box
    diagonal, whereas line/arc fragments have a ratio near one.
    """
    skel = skeletonize(mask > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(skel, 8)
    chains = []
    ys, xs = np.nonzero(skel)
    labels = lab[ys, xs]
    diag = float(np.hypot(*mask.shape))
    for i in range(1, n):
        sel = labels == i
        if sel.sum() < min_px:
            continue
        pts = np.stack([xs[sel], ys[sel]], axis=1).astype(float)
        bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if bbox_diag < 0.10 * diag and len(pts) > 2.2 * max(bbox_diag, 1.0):
            continue  # compact dense loop: player / blob outline
        chains.append(pts)
    return chains


# ---------------------------------------------------------------------- #
# Distortion self-calibration (division model)
# ---------------------------------------------------------------------- #
def _straightness(pts: np.ndarray) -> float:
    """RMS perpendicular distance of points to their total-LS line."""
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q / len(q)
    evals = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(evals[0], 0.0)))


def _long_strips(
    chains: list[np.ndarray], diag: float, rng: np.random.Generator
) -> list[tuple[np.ndarray, float]]:
    """Extract long, roughly-line-like point strips for distortion fitting.

    Uses a generous RANSAC tolerance so that *distorted* (gently curved)
    straight lines are still captured as single strips, while corners of
    composite chains are split apart.
    """
    strips: list[tuple[np.ndarray, float]] = []
    min_extent = 0.12 * diag
    tol = 2.5
    for ch in chains:
        pts = ch[:: max(1, len(ch) // 800)].copy()
        for _ in range(3):
            if len(pts) < 50:
                break
            best: np.ndarray | None = None
            for _ in range(120):
                idx = rng.choice(len(pts), 3, replace=False)
                p1, p2, p3 = pts[idx]
                if np.linalg.norm(p3 - p1) < min_extent * 0.5:
                    continue
                # Circumcircle through 3 points; falls back to a line when
                # they are (near-)collinear — a straight pitch line, or a
                # distorted one, is a circular arc in the image.
                a = p2 - p1
                b = p3 - p1
                denom = 2.0 * (a[0] * b[1] - a[1] * b[0])
                if abs(denom) < 1e-6 * np.linalg.norm(a) * np.linalg.norm(b):
                    d = p3 - p1
                    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                    resid = np.abs((pts - p1) @ nrm)
                else:
                    ux = (b[1] * (a @ a) - a[1] * (b @ b)) / denom
                    uy = (a[0] * (b @ b) - b[0] * (a @ a)) / denom
                    c = p1 + [ux, uy]
                    r = np.linalg.norm([ux, uy])
                    resid = np.abs(np.linalg.norm(pts - c, axis=1) - r)
                inl = resid < tol
                if best is None or inl.sum() > best.sum():
                    best = inl
            if best is None or best.sum() < 50:
                break
            sel = pts[best]
            extent = float(np.linalg.norm(sel.max(axis=0) - sel.min(axis=0)))
            # Reject sparse collinear junk: a real marking strip covers its
            # span roughly uniformly, accidental alignments come in clumps
            # separated by long gaps.
            c = sel.mean(axis=0)
            _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
            t = np.sort((sel - c) @ vt[0])
            max_gap = float(np.diff(t).max()) if len(t) > 1 else extent
            if extent >= min_extent and max_gap < 0.25 * extent:
                strips.append((sel, extent))
            pts = pts[~best]
    return strips


def estimate_distortion(
    chains: list[np.ndarray],
    image_size: tuple[int, int],
    lam_range: tuple[float, float] = (-0.08, 0.45),
    steps: int = 54,
    rng: np.random.Generator | None = None,
) -> DivisionModel:
    """Estimate a division model that straightens long marking strips.

    Strips that stay curved for every parameter choice (centre-circle arcs)
    hit a saturating extent-proportional cap and do not bias the estimate.
    A centred single-parameter model is tried first; with enough
    independent strip evidence a free-centre three-parameter fit is also
    tried (virtual-PTZ crops of panoramic rigs have off-centre
    distortion), guarded against overfit by requiring at least three
    strips, a plausible centre, and a decisive improvement.  Returns the
    identity model when the frame shows no meaningful curvature.
    """
    rng = rng or np.random.default_rng(11)
    w, h = image_size
    diag = float(np.hypot(w, h))
    identity = DivisionModel(0.0, image_size)
    strips = _long_strips(chains, diag, rng)
    if not strips:
        return identity

    def cost_dm(dm: DivisionModel) -> float:
        tot, wsum = 0.0, 0.0
        for pts, extent in strips:
            r = _straightness(dm.undistort(pts))
            # Extent-proportional cap: keeps gradient alive for strongly
            # distorted straight lines (sagitta grows with length) while
            # still saturating on genuine arcs (circle segments).
            tot += min(r, max(6.0, 0.03 * extent)) * extent
            wsum += extent
        return tot / wsum

    c0 = cost_dm(identity)
    if c0 < 1.5:  # no meaningful curvature to explain
        return identity

    # 1) centred one-parameter grid + parabolic refinement.
    lams = np.linspace(lam_range[0], lam_range[1], steps)
    costs = np.array([cost_dm(DivisionModel(l, image_size)) for l in lams])
    i = int(np.argmin(costs))
    lam_best = float(lams[i])
    if 0 < i < steps - 1:
        y0, y1, y2 = costs[i - 1 : i + 2]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lam_best += 0.5 * (y0 - y2) / denom * (lams[1] - lams[0])
    best = DivisionModel(lam_best, image_size)
    cb = cost_dm(best)
    if cb > 0.7 * c0:  # improvement not decisive: assume no distortion
        return identity
    return best


# ---------------------------------------------------------------------- #
# Straight-line extraction from chains
# ---------------------------------------------------------------------- #
def _fit_line_tls(pts: np.ndarray) -> np.ndarray:
    """Total-least-squares homogeneous line through points."""
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q
    evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, 0]  # normal = smallest-eigenvalue direction
    return normalize_line(np.array([n[0], n[1], -n @ c]))


def _line_piece_ok(
    sel: np.ndarray, line: np.ndarray, min_extent: float
) -> tuple[bool, bool]:
    """Quality tests for a candidate straight piece.

    Returns (accept_as_line, keep_as_curve).  Rejects (a) short pieces,
    (b) sparse or gappy collinear runs (goal nets, haze, scattered noise --
    real paint lines have near-contiguous ~1 px/px skeleton support), and
    (c) arcs: if a circle fits the points substantially better than the
    line, this is a low-sagitta chord of the centre circle / penalty arc.
    """
    proj = sel @ np.array([line[1], -line[0]])
    order = np.argsort(proj)
    t = proj[order]
    span = float(t[-1] - t[0])
    if span < min_extent:
        return False, True
    density = len(sel) / span
    max_gap = float(np.diff(t).max())
    if density < 0.45 or max_gap > 0.45 * span:
        return False, False  # junk: do not feed the ellipse stage either
    # Quadratic sagitta test: a residual curve trend means an arc chord.
    r = np.column_stack([sel, np.ones(len(sel))]) @ line
    tc = proj - proj.mean()
    A = np.stack([tc**2, tc, np.ones_like(tc)], axis=1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    sagitta = abs(coef[0]) * (span / 2.0) ** 2
    if sagitta > max(2.0, 0.006 * span):
        return False, True
    return True, False


def _chain_lines(
    chain: np.ndarray, rng: np.random.Generator, diag: float
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Extract straight pieces from one chain via sequential RANSAC.

    Returns (list of (line, inlier_points), leftover_points).
    """
    pts = chain.copy()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    curved: list[np.ndarray] = []
    tol = 2.2
    min_extent = max(55.0, 0.045 * diag)
    for _ in range(4):
        if len(pts) < 45:
            break
        best_in: np.ndarray | None = None
        n_iter = 64
        for _ in range(n_iter):
            i, j = rng.choice(len(pts), 2, replace=False)
            if np.linalg.norm(pts[i] - pts[j]) < min_extent * 0.5:
                continue
            d = pts[j] - pts[i]
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            dist = np.abs((pts - pts[i]) @ nrm)
            inl = dist < tol
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
        if best_in is None or best_in.sum() < 45:
            break
        line = _fit_line_tls(pts[best_in])
        dist = np.abs(np.column_stack([pts, np.ones(len(pts))]) @ line)
        inl = dist < tol
        if inl.sum() < 45:
            break
        sel = pts[inl]
        ok, is_curved = _line_piece_ok(sel, line, min_extent)
        if ok:
            out.append((line, sel))
        elif is_curved:
            curved.append(sel)  # keep for the ellipse stage
        pts = pts[~inl]
    leftover = pts
    if curved:
        leftover = np.concatenate([leftover] + curved) if len(leftover) else np.concatenate(curved)
    return out, leftover


def merge_lines(
    raw: list[tuple[np.ndarray, np.ndarray]], diag: float
) -> list[DetectedLine]:
    """Merge near-collinear pieces (dashed / occlusion-broken lines)."""
    used = [False] * len(raw)
    merged: list[DetectedLine] = []
    for i in range(len(raw)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        li, pi = raw[i]
        for j in range(i + 1, len(raw)):
            if used[j]:
                continue
            lj, pj = raw[j]
            ang = np.arccos(np.clip(abs(li[0] * lj[0] + li[1] * lj[1]), 0, 1))
            if ang > np.deg2rad(2.0):
                continue
            # Mutual perpendicular distance of the other group's points.
            dij = np.abs(np.column_stack([pj, np.ones(len(pj))]) @ li).mean()
            dji = np.abs(np.column_stack([pi, np.ones(len(pi))]) @ lj).mean()
            if max(dij, dji) < max(5.0, 0.004 * diag):
                group.append(j)
                used[j] = True
        pts = np.concatenate([raw[g][1] for g in group])
        line = _fit_line_tls(pts)
        d = np.array([line[1], -line[0]])
        t = pts @ d
        p0 = pts[int(np.argmin(t))]
        p1 = pts[int(np.argmax(t))]
        merged.append(DetectedLine(l=line, p0=p0, p1=p1, support=len(pts), points=pts))
    merged.sort(key=lambda m: -m.support)
    return merged


# ---------------------------------------------------------------------- #
# Ellipse (centre circle / penalty arc)
# ---------------------------------------------------------------------- #
def _sampson_dist(conic: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """First-order geometric distance of points to a conic."""
    ph = np.column_stack([pts, np.ones(len(pts))])
    val = np.einsum("ni,ij,nj->n", ph, conic, ph)
    grad = 2.0 * (ph @ conic)[:, :2]
    gn = np.linalg.norm(grad, axis=1)
    return np.abs(val) / np.maximum(gn, 1e-9)


def _tangent_diversity_deg(conic: np.ndarray, pts: np.ndarray) -> float:
    """Spread (deg) of conic tangent directions at the given points.

    A genuine ellipse arc curves through many tangent directions; inliers
    of a degenerate "ellipse" hugging a straight line share one direction.
    Measured as occupied 10-degree bins of the normal angle modulo pi.
    """
    if len(pts) == 0:
        return 0.0
    ph = np.column_stack([pts, np.ones(len(pts))])
    grad = (ph @ conic)[:, :2]
    ang = np.mod(np.arctan2(grad[:, 1], grad[:, 0]), np.pi)
    idx = (ang / np.pi * 18).astype(int) % 18
    counts = np.bincount(idx, minlength=18)
    # Only bins holding real mass count -- a straight line grazing the
    # high-curvature vertex of a degenerate ellipse leaves a trickle of
    # points across many bins.
    occupied = counts >= max(2, int(0.02 * len(pts)))
    return float(occupied.sum() * 10.0)


def _ellipse_from_points(pts: np.ndarray, diag: float) -> DetectedEllipse | None:
    """Iteratively-refit ellipse on a point set; None if degenerate."""
    from .geometry import ellipse_to_conic

    for tol in (8.0, 4.0, 2.5):
        if len(pts) < 40:
            return None
        fit = cv2.fitEllipseDirect(pts.astype(np.float32))
        (cx, cy), (w2, h2), ang = fit
        if not np.isfinite([cx, cy, w2, h2, ang]).all() or min(w2, h2) < 4:
            return None
        a, b = max(w2, h2) / 2.0, min(w2, h2) / 2.0
        if a > 3.0 * diag:
            return None
        th = np.deg2rad(ang if w2 >= h2 else ang + 90.0)
        conic = ellipse_to_conic(cx, cy, a, b, th)
        pts = pts[_sampson_dist(conic, pts) < tol]
    if len(pts) < 60 or b < 15 or a < 50 or a > 1.5 * diag or a / b > 12:
        return None
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.unique((np.floor((angles + np.pi) / (2 * np.pi) * 36)).astype(int))
    span = float(len(bins) * 10.0)
    # Supporting pixels must plausibly cover the claimed angular span of
    # the perimeter (skeletons are ~1 px thick), else this is a junk fit
    # of scattered points around a small ellipse.
    perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    if len(pts) < 0.35 * perimeter * (span / 360.0):
        return None
    if _tangent_diversity_deg(conic, pts) < 60.0:
        return None
    kind = "circle" if span >= 150.0 else "arc"
    return DetectedEllipse(
        conic=conic,
        center=np.array([cx, cy]),
        axes=(a, b),
        angle_deg=float(ang),
        span_deg=span,
        support=len(pts),
        kind=kind,
        points=pts,
    )


def _arc_strips(
    chains: list[np.ndarray], diag: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Genuinely curved strips: circle-RANSAC pieces with clear sagitta."""
    strips: list[np.ndarray] = []
    tol = 2.0
    min_extent = 0.05 * diag
    for ch in chains:
        pts = ch[:: max(1, len(ch) // 800)].copy()
        for _ in range(4):
            if len(pts) < 35:
                break
            best: np.ndarray | None = None
            best_r = 0.0
            for _ in range(150):
                idx = rng.choice(len(pts), 3, replace=False)
                p1, p2, p3 = pts[idx]
                a = p2 - p1
                b = p3 - p1
                denom = 2.0 * (a[0] * b[1] - a[1] * b[0])
                if abs(denom) < 1e-9:
                    continue
                ux = (b[1] * (a @ a) - a[1] * (b @ b)) / denom
                uy = (a[0] * (b @ b) - b[0] * (a @ a)) / denom
                c = p1 + [ux, uy]
                r = float(np.hypot(ux, uy))
                if r > 2.5 * diag:
                    continue
                inl = np.abs(np.linalg.norm(pts - c, axis=1) - r) < tol
                if best is None or inl.sum() > best.sum():
                    best = inl
                    best_r = r
            if best is None or best.sum() < 35:
                break
            sel = pts[best]
            extent = float(np.linalg.norm(sel.max(axis=0) - sel.min(axis=0)))
            sagitta = extent**2 / (8.0 * max(best_r, 1e-6))
            if extent >= min_extent and sagitta > 3.5:
                strips.append(sel)
            pts = pts[~best]
    return strips


def detect_ellipse_ransac(
    chains: list[np.ndarray],
    diag: float,
    rng: np.random.Generator,
) -> DetectedEllipse | None:
    """Find the dominant ellipse (centre circle / penalty arc).

    Runs *before* line extraction, because chords of a large projected
    circle are locally indistinguishable from straight lines -- the circle
    must be claimed globally first.  Hypotheses are ellipses fitted to
    unions of genuinely curved strips (single strips, pairs and triples, so
    occlusion-fragmented circles are recovered), scored by Sampson-distance
    inliers over all marking points, then iteratively refit.
    """
    from .geometry import ellipse_to_conic

    if not chains:
        return None
    pool = np.concatenate([c for c in chains if len(c) >= 20])
    if len(pool) < 120:
        return None
    sub = pool[:: max(1, len(pool) // 6000)]

    strips = _arc_strips(chains, diag, rng)
    strips.sort(key=len, reverse=True)
    strips = strips[:8]
    subsets: list[np.ndarray] = list(strips)
    for i in range(len(strips)):
        for j in range(i + 1, len(strips)):
            subsets.append(np.concatenate([strips[i], strips[j]]))
    for i in range(min(4, len(strips))):
        for j in range(i + 1, min(4, len(strips))):
            for k in range(j + 1, min(4, len(strips))):
                subsets.append(np.concatenate([strips[i], strips[j], strips[k]]))
    # Local five-point hypotheses: strips can all be junk arcs (accidental
    # collinear-ish runs across different lines), so a second, independent
    # hypothesis source samples small windows of the point pool directly.
    for _ in range(250):
        anchor = pool[rng.integers(len(pool))]
        radius = diag * rng.uniform(0.04, 0.35)
        d = np.linalg.norm(sub - anchor, axis=1)
        window = sub[d < radius]
        if len(window) >= 24:
            subsets.append(window[rng.choice(len(window), 5, replace=False)])
    if not subsets:
        return None

    best_inl: np.ndarray | None = None
    best_n = 0
    for pts in subsets:
        if len(pts) < 6:
            continue
        fit = cv2.fitEllipseDirect(pts.astype(np.float32))
        (cx, cy), (w2, h2), ang = fit
        if not np.isfinite([cx, cy, w2, h2, ang]).all():
            continue
        a, b = max(w2, h2) / 2.0, min(w2, h2) / 2.0
        if b < 15 or a < 50 or a > 1.5 * diag or a / max(b, 1e-6) > 12:
            continue
        th = np.deg2rad(ang if w2 >= h2 else ang + 90.0)
        conic = ellipse_to_conic(cx, cy, a, b, th)
        inl = _sampson_dist(conic, sub) < 2.5
        n = int(inl.sum())
        if n > best_n and _tangent_diversity_deg(conic, sub[inl]) >= 60.0:
            best_n = n
            best_inl = inl
    if best_inl is None or best_n < 60:
        return None
    cand = _ellipse_from_points(sub[best_inl], diag)
    if cand is None:
        return None
    # Re-collect inliers from the full-resolution pool and refit, a few
    # annealed rounds: a partial-arc initial fit extrapolates the far side
    # poorly, so start generous and tighten.
    for tol in (10.0, 6.0, 3.0, 2.5):
        full = pool[_sampson_dist(cand.conic, pool) < tol]
        cand = _ellipse_from_points(full, diag) or cand
    return cand


# ---------------------------------------------------------------------- #
# Top-level detection
# ---------------------------------------------------------------------- #
def detect_features(
    bgr: np.ndarray,
    max_proc_width: int = 1400,
    rng: np.random.Generator | None = None,
    player_boxes: np.ndarray | None = None,
) -> DetectedFeatures:
    """Run the full detection stack on a BGR frame.

    Args:
        bgr: input frame.
        max_proc_width: mask/skeleton processing resolution.
        rng: RANSAC randomness (fixed default: deterministic results).
        player_boxes: optional (N, 4) player boxes as (x0, y0, x1, y1) in
            pixel coords, e.g. from the upstream detection stage.  Masking
            them out removes white-kit edges from the marking evidence.
    """
    rng = rng or np.random.default_rng(3)
    h0, w0 = bgr.shape[:2]
    scale = 1.0
    img = bgr
    if w0 > max_proc_width:
        scale = max_proc_width / w0
        img = cv2.resize(bgr, (max_proc_width, int(round(h0 * scale))))
    field = field_mask(img)
    field_frac = float((field > 0).mean())
    marks = marking_mask(img, field)
    if player_boxes is not None:
        for x0, y0, x1, y1 in np.asarray(player_boxes, dtype=float):
            pad_x = 0.1 * (x1 - x0)
            pad_y = 0.05 * (y1 - y0)
            a = np.array([x0 - pad_x, y0 - pad_y]) * scale
            b = np.array([x1 + pad_x, y1 + pad_y]) * scale
            marks[
                max(0, int(a[1])) : max(0, int(np.ceil(b[1]))),
                max(0, int(a[0])) : max(0, int(np.ceil(b[0]))),
            ] = 0
    chains_s = skeleton_chains(marks)
    # Back to original pixel coordinates.
    chains = [c / scale for c in chains_s]
    diag = float(np.hypot(w0, h0))

    dm = estimate_distortion(chains, (w0, h0))
    lam = dm.lam
    und_chains = [dm.undistort(c) for c in chains]

    # Ellipse first: chords of a large projected circle look locally
    # straight, so the circle must be claimed before line extraction.
    ellipse = detect_ellipse_ransac(und_chains, diag, rng)
    line_chains = und_chains
    if ellipse is not None:
        line_chains = []
        for ch in und_chains:
            keep = ch[_sampson_dist(ellipse.conic, ch) > 3.5]
            if len(keep) >= 25:
                line_chains.append(keep)

    raw_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for ch in line_chains:
        pieces, _left = _chain_lines(ch, rng, diag)
        raw_lines.extend(pieces)
    lines = merge_lines(raw_lines, diag)
    lines = [ln for ln in lines if ln.length > 0.05 * diag]

    all_pts = (
        np.concatenate(und_chains) if und_chains else np.zeros((0, 2))
    )
    # Working canvas covering all undistorted content.
    corners = dm.undistort(
        np.array([[0, 0], [w0, 0], [w0, h0], [0, h0], [w0 / 2, 0], [w0 / 2, h0]])
    )
    lo = np.floor(np.minimum(corners.min(axis=0), [0, 0])).astype(int)
    hi = np.ceil(np.maximum(corners.max(axis=0), [w0, h0])).astype(int)
    return DetectedFeatures(
        lines=lines,
        ellipse=ellipse,
        lam=lam,
        dm=dm,
        line_points=all_pts,
        image_size=(w0, h0),
        canvas_origin=lo.astype(float),
        canvas_size=(int(hi[0] - lo[0]), int(hi[1] - lo[1])),
        field_fraction=field_frac,
        diagnostics={"n_chains": len(chains), "n_raw_lines": len(raw_lines)},
    )
