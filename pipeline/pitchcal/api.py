"""Public API: calibrate a single broadcast frame to pitch coordinates.

Example
-------
>>> import cv2
>>> from pitchcal import calibrate, CalibrationStatus
>>> frame = cv2.imread("frame.png")
>>> result = calibrate(frame)
>>> if result.ok:
...     pitch_xy = result.image_to_pitch([[960, 540]])   # metres
... else:
...     print(result.status, result.diagnostics)

Coordinates: x in [0, 105] along the touchlines, y in [0, 68] along the
goal lines, metres.  Canonical orientation puts the camera on the y < 0
side (the touchline nearest the camera is y = 0).  Which stadium end is
x = 0 is unknowable from a single frame's geometry -- the canonical frame
is *consistent per frame*, not absolutely oriented.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import DetectedFeatures, detect_features
from .geometry import DivisionModel, apply_h_w, canonicalize
from .orientation import validate_and_canonicalize_homography
from .pitch_model import PitchModel
from .players import formation_lines
from .solve import (
    Candidate,
    Scorer,
    enumerate_circle_candidates,
    enumerate_formation_candidates,
    enumerate_line_candidates,
    mapping_discrepancy_m,
    refine_candidate,
)


class CalibrationStatus(enum.Enum):
    """Outcome of a calibration attempt."""

    OK = "ok"
    INSUFFICIENT_FEATURES = "insufficient_features"  # geometry under-determined
    NO_CONSENSUS = "no_consensus"  # candidates exist but none fits the evidence
    AMBIGUOUS = "ambiguous"  # two structurally different fits are both plausible
    LOW_CONFIDENCE = "low_confidence"  # best fit passed gates but is not trustworthy
    NO_PITCH = "no_pitch"  # frame does not look at a pitch at all


@dataclass
class CalibrationResult:
    """A pixel <-> pitch mapping plus quality metadata.

    ``H`` maps pitch coordinates (metres, homogeneous) to *undistorted*
    pixel coordinates; lens distortion is handled by ``dm``.  Use
    :meth:`image_to_pitch` / :meth:`pitch_to_image` which compose both.
    """

    status: CalibrationStatus
    H: np.ndarray | None = None  # pitch -> undistorted image (3x3)
    dm: DivisionModel | None = None
    image_size: tuple[int, int] = (0, 0)
    confidence: float = 0.0  # [0, 1]
    fit_px: float = float("nan")  # mean marking distance of template, px
    explained: float = 0.0  # fraction of observed marking pixels explained
    used_player_formation: bool = False
    flipped_for_canonical: bool = False  # rot180 applied by canonicalisation
    pitch: PitchModel = field(default_factory=PitchModel)
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is CalibrationStatus.OK

    def image_to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        """Map (N, 2) pixel coords to (N, 2) pitch coords in metres."""
        if self.H is None or self.dm is None:
            raise ValueError(f"calibration unavailable: {self.status}")
        und = self.dm.undistort(np.asarray(pts_px, dtype=float))
        p, _ = apply_h_w(np.linalg.inv(self.H), und)
        return p

    def pitch_to_image(self, pts_m: np.ndarray) -> np.ndarray:
        """Map (N, 2) pitch coords in metres to (N, 2) pixel coords."""
        if self.H is None or self.dm is None:
            raise ValueError(f"calibration unavailable: {self.status}")
        p, _ = apply_h_w(self.H, np.asarray(pts_m, dtype=float))
        return self.dm.distort(p)

    def warp_topdown(self, image: np.ndarray, px_per_m: float = 10.0) -> np.ndarray:
        """Render the frame onto a top-down pitch map (for inspection)."""
        if self.H is None or self.dm is None:
            raise ValueError(f"calibration unavailable: {self.status}")
        L, W = self.pitch.length, self.pitch.width
        out_w, out_h = int(L * px_per_m), int(W * px_per_m)
        gx, gy = np.meshgrid(np.arange(out_w) / px_per_m, np.arange(out_h) / px_per_m)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        img_pts = self.pitch_to_image(pts).astype(np.float32)
        map_x = img_pts[:, 0].reshape(out_h, out_w)
        map_y = img_pts[:, 1].reshape(out_h, out_w)
        return cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderValue=(30, 30, 30)
        )


def _confidence(best: Candidate, margin: float, n_lines: int) -> float:
    """Heuristic [0, 1] confidence from fit quality and solution margin."""
    c = 1.0
    c *= float(np.clip((0.30 - best.fit) / 0.25, 0.0, 1.0))
    c *= float(np.clip((best.explained - 0.50) / 0.35, 0.0, 1.0))
    c *= float(np.clip(margin / 2.0, 0.3, 1.0))
    # A tiny visible-template footprint makes the chamfer fit unreliable.
    c *= float(np.clip((best.n_inside - 50) / 150.0, 0.5, 1.0))
    if best.used_players:
        c *= 0.6
    if best.n_constraints <= 8:
        c *= 0.85
    return float(np.clip(c, 0.0, 1.0))


def calibrate(
    image_bgr: np.ndarray,
    player_feet: np.ndarray | None = None,
    player_labels: np.ndarray | None = None,
    player_boxes: np.ndarray | None = None,
    pitch: PitchModel | None = None,
    max_hyp: int = 25000,
    accept_fit: float = 0.40,
    accept_explained: float = 0.45,
    min_confidence: float = 0.30,
    ambiguity_m: float = 3.0,
    ambiguity_margin: float = 1.20,
) -> CalibrationResult:
    """Calibrate a single broadcast frame.

    Args:
        image_bgr: the frame (BGR uint8, any resolution).
        player_feet: optional (N, 2) player ground points in pixel coords
            (e.g. bbox bottom-centres from the detection stage).  Enables
            the formation-line fallback for marking-sparse frames.
        player_labels: optional (N,) team labels aligned with player_feet.
        player_boxes: optional (N, 4) player boxes (x0, y0, x1, y1); their
            content is masked out of the marking evidence, which removes
            white-kit edges that otherwise pollute detection and scoring.
        pitch: pitch dimensions override (defaults to 105 x 68 FIFA).
        max_hyp: budget for the assignment search.
        accept_fit: max normalised chamfer fit for an OK result.
        accept_explained: min explained fraction for an OK result.
        ambiguity_m: mapping discrepancy (metres) above which a close
            runner-up marks the result AMBIGUOUS.
        ambiguity_margin: score ratio under which a runner-up is "close".

    Returns:
        A :class:`CalibrationResult`; ``result.ok`` tells whether ``H`` is
        trustworthy.  A failed result never contains a homography.
    """
    t0 = time.time()
    pitch = pitch or PitchModel()
    h, w = image_bgr.shape[:2]

    features = detect_features(image_bgr, player_boxes=player_boxes)
    diag: dict = {
        "n_lines": len(features.lines),
        "ellipse": None
        if features.ellipse is None
        else {
            "kind": features.ellipse.kind,
            "span_deg": features.ellipse.span_deg,
            "support": features.ellipse.support,
        },
        "lambda": features.lam,
        "field_fraction": features.field_fraction,
    }

    if features.field_fraction < 0.08:
        return CalibrationResult(
            status=CalibrationStatus.NO_PITCH,
            image_size=(w, h),
            pitch=pitch,
            diagnostics=diag | {"reason": "no significant grass area"},
        )

    scorer = Scorer(features, pitch)
    candidates: list[Candidate] = []
    candidates += enumerate_line_candidates(features, pitch, scorer, max_hyp=max_hyp)
    candidates += enumerate_circle_candidates(features, pitch, scorer)

    # Player-formation fallback: only when paint alone looks insufficient.
    have_good_paint = any(c.fit < 0.22 for c in candidates)
    if player_feet is not None and not have_good_paint:
        und_feet = features.dm.undistort(np.asarray(player_feet, dtype=float))
        flines = formation_lines(und_feet, player_labels, (w, h))
        candidates += enumerate_formation_candidates(features, pitch, scorer, flines)

    diag["n_candidates"] = len(candidates)
    if not candidates:
        return CalibrationResult(
            status=CalibrationStatus.INSUFFICIENT_FEATURES,
            image_size=(w, h),
            pitch=pitch,
            diagnostics=diag | {"elapsed_s": time.time() - t0},
        )

    # Shortlist by chamfer fit, add the explained-coverage term, refine.
    candidates.sort(key=lambda c: c.fit)
    shortlist = candidates[:24]
    for c in shortlist:
        c.explained = scorer.explained(c.H)
    shortlist.sort(key=lambda c: c.score)
    refined = [refine_candidate(c, scorer, features) for c in shortlist[:6]]
    for c in refined:
        c.explained = scorer.explained(c.H)
    refined.sort(key=lambda c: c.score)
    best = refined[0]

    worst_line = scorer.worst_line_explained(
        best.H,
        features.lines,
        ellipse_points=None if features.ellipse is None else features.ellipse.points,
    )
    diag["best"] = {
        "strategy": best.strategy,
        "worst_line_explained": round(worst_line, 3),
        "fit": round(best.fit, 4),
        "explained": round(best.explained, 3),
        "assignment": {int(k): v for k, v in best.assignment.items()},
        "n_constraints": best.n_constraints,
    }
    diag["elapsed_s"] = round(time.time() - t0, 2)

    if best.fit > accept_fit or best.explained < accept_explained:
        return CalibrationResult(
            status=CalibrationStatus.NO_CONSENSUS,
            image_size=(w, h),
            pitch=pitch,
            diagnostics=diag | {"reason": "best candidate fits the evidence poorly"},
        )
    if worst_line < 0.30:
        return CalibrationResult(
            status=CalibrationStatus.NO_CONSENSUS,
            image_size=(w, h),
            pitch=pitch,
            diagnostics=diag
            | {"reason": "a major detected line is unexplained by the best candidate"},
        )

    # Validate + canonicalise (camera on y < 0 side) via the standalone
    # orientation checker, *before* comparing alternatives, so rot180 twins
    # collapse into one solution.  Validity cannot fail here in practice --
    # every surviving candidate already passed the same proper-view test
    # during solving/refinement -- but stay defensive.
    check = validate_and_canonicalize_homography(best.H, image_bgr, pitch=pitch)
    if not check.is_valid:  # pragma: no cover - upstream filters prevent this
        return CalibrationResult(
            status=CalibrationStatus.NO_CONSENSUS,
            image_size=(w, h),
            pitch=pitch,
            diagnostics=diag | {"reason": f"orientation check: {check.reason}"},
        )
    H_best, flipped = check.corrected_H, check.was_flipped

    margin = 10.0
    runner = None
    for c in refined[1:]:
        Hc, _ = canonicalize(c.H, pitch.length, pitch.width)
        if mapping_discrepancy_m(H_best, Hc, (w, h)) > ambiguity_m:
            runner = c
            break
    if runner is not None and best.score > 1e-9:
        margin = runner.score / max(best.score, 1e-9)
        if margin < ambiguity_margin and runner.explained > accept_explained:
            diag["runner_up"] = {
                "strategy": runner.strategy,
                "fit": round(runner.fit, 4),
                "explained": round(runner.explained, 3),
                "score_ratio": round(margin, 3),
            }
            return CalibrationResult(
                status=CalibrationStatus.AMBIGUOUS,
                image_size=(w, h),
                pitch=pitch,
                diagnostics=diag,
            )

    confidence = _confidence(best, margin, len(features.lines))
    if confidence < min_confidence:
        return CalibrationResult(
            status=CalibrationStatus.LOW_CONFIDENCE,
            image_size=(w, h),
            confidence=confidence,
            pitch=pitch,
            diagnostics=diag
            | {"reason": f"confidence {confidence:.2f} below {min_confidence}"},
        )

    return CalibrationResult(
        status=CalibrationStatus.OK,
        H=H_best,
        dm=features.dm,
        image_size=(w, h),
        confidence=confidence,
        fit_px=best.fit * scorer.cap / scorer.scale,
        explained=best.explained,
        used_player_formation=best.used_players,
        flipped_for_canonical=flipped,
        pitch=pitch,
        diagnostics=diag,
    )


def render_overlay(image: np.ndarray, result: CalibrationResult) -> np.ndarray:
    """Draw the calibrated template back onto the frame (for inspection)."""
    vis = image.copy()
    if not result.ok:
        cv2.putText(
            vis,
            f"CALIBRATION FAILED: {result.status.value}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        return vis
    for _, pts in result.pitch.sample_all(step=0.35):
        proj = result.pitch_to_image(pts)
        _, wz = apply_h_w(result.H, pts)
        p = np.round(proj).astype(int)
        h, w = vis.shape[:2]
        for i in range(len(p) - 1):
            if wz[i] > 1e-9 and wz[i + 1] > 1e-9:
                a, b = p[i], p[i + 1]
                if (
                    -w < a[0] < 2 * w
                    and -h < a[1] < 2 * h
                    and np.abs(b - a).max() < 100
                ):
                    cv2.line(vis, tuple(a), tuple(b), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"conf={result.confidence:.2f} fit={result.fit_px:.1f}px expl={result.explained:.2f}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 255),
        2,
    )
    return vis
