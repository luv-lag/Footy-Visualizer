"""Standalone orientation validation / canonicalization for ANY homography.

The pitch's marking layout is symmetric under {identity, mirror-x, mirror-y,
rot180}, so a calibration system matching markings to a template can produce
a geometrically-accurate homography in the wrong orientation -- the
"near/far goal swapped" failure mode.  pitchcal resolves this internally for
its own candidates; this module exposes the same two physical checks as a
self-contained function usable on homographies produced by *any* system
(e.g. PnLCalib), with no dependence on pitchcal's detection pipeline:

1. **Handedness (mirror rejection).**  For any real camera above the pitch
   producing a non-mirrored image (v-axis down), the pitch->image mapping is
   orientation-REVERSING: ``sign(det H) * sign(w)^3 < 0`` at every pitch
   point in front of the camera.  A homography containing an odd number of
   mirror symmetries violates this and is physically impossible -- it is
   rejected, not corrected (composing a mirror would silently change the
   geometry an external system committed to).

2. **Perspective asymmetry (rot180 canonicalization).**  Pitch geometry is
   180-degree symmetric but camera perspective is not: the touchline nearer
   the camera projects with larger local scale (|det J| = |det H| / |w|^3,
   i.e. smaller |w|).  The output is canonicalized so the camera sits on
   the y < 0 side of the pitch (near touchline = y = 0).  Which physical
   stadium end is x = 0 is unknowable from one frame; canonical output is
   *consistent*, which is what eliminates near/far-swap errors.

Scope: orientation and physical validity ONLY.  A homography that is merely
inaccurate (bad correspondences, wrong scale) but properly oriented passes
through unchanged; detecting that is out of scope here.

Both checks need only the homography and the frame's pixel dimensions; the
image is accepted (rather than a size tuple) for call-site convenience and
future-proofing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import canonicalize, proper_view_violation
from .pitch_model import PitchModel


@dataclass(frozen=True)
class CalibrationCheck:
    """Outcome of validating/canonicalizing an externally-supplied homography.

    Attributes:
        corrected_H: the canonicalized pitch->image homography (Frobenius-
            normalized; homographies are scale-equivalent) if the input was
            physically valid, else ``None``.  Uses the same pitch-coordinate
            convention as the input (corner-origin unless ``origin="center"``
            was passed, in which case the OUTPUT is corner-origin -- see
            :func:`validate_and_canonicalize_homography`).
        is_valid: False iff the input was rejected as physically impossible
            (mirror-image / degenerate), in which case ``corrected_H`` is
            ``None``.
        was_flipped: True iff the 180-degree correction was actually applied.
            False when the input was already canonically oriented (or
            invalid).
        reason: human-readable explanation -- why the input was rejected,
            what was corrected and why, or confirmation that it was already
            correctly oriented.
    """

    corrected_H: np.ndarray | None
    is_valid: bool
    was_flipped: bool
    reason: str


def validate_and_canonicalize_homography(
    H: np.ndarray,
    image: np.ndarray,
    pitch: PitchModel | None = None,
    origin: str = "corner",
) -> CalibrationCheck:
    """Validate physical plausibility of ``H`` and canonicalize orientation.

    Designed for homographies produced by ANY calibration system: the only
    inputs used are the matrix itself and the frame's dimensions.  See the
    module docstring for the two checks performed and their physics.

    Args:
        H: (3, 3) homography mapping pitch ground-plane coordinates in
            metres to image pixel coordinates, ``[u, v, w]^T ~ H [x, y, 1]^T``.
        image: the source frame (only ``image.shape`` is used, to bound
            which pitch points count as visible).
        pitch: pitch dimensions; defaults to FIFA 105 x 68.
        origin: pitch-coordinate convention of the INPUT matrix:
            ``"corner"`` -- origin at a pitch corner, x in [0, length],
            y in [0, width] (pitchcal's convention); ``"center"`` -- origin
            at the pitch centre, x in [-length/2, length/2] (PnLCalib's
            convention).  The returned ``corrected_H`` is ALWAYS in
            corner-origin convention, which is what canonical orientation
            is defined in.

    Returns:
        A :class:`CalibrationCheck`.  ``corrected_H`` is ``None`` iff
        ``is_valid`` is False.
    """
    pitch = pitch or PitchModel()
    length, width = pitch.length, pitch.width

    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        return CalibrationCheck(
            None, False, False, f"expected a 3x3 homography, got shape {H.shape}"
        )
    if not np.isfinite(H).all():
        return CalibrationCheck(None, False, False, "homography contains non-finite values")
    if abs(np.linalg.det(H / max(np.abs(H).max(), 1e-300))) < 1e-12:
        return CalibrationCheck(None, False, False, "homography is (near-)singular")

    if origin == "center":
        # (x, y) corner-origin -> (x - L/2, y - W/2) centre-origin.
        T = np.array(
            [[1.0, 0.0, -length / 2.0], [0.0, 1.0, -width / 2.0], [0.0, 0.0, 1.0]]
        )
        H = H @ T
    elif origin != "corner":
        return CalibrationCheck(
            None, False, False, f'origin must be "corner" or "center", got {origin!r}'
        )
    H = H / np.linalg.norm(H)

    h, w = image.shape[:2]
    violation = proper_view_violation(H, length, width, (float(w), float(h)))
    if violation is not None:
        return CalibrationCheck(None, False, False, f"rejected: {violation}")

    H_canon, flipped = canonicalize(H, length, width)
    if flipped:
        reason = (
            "180-degree orientation corrected: the touchline the input "
            "placed at y=width projects with larger local scale (it is the "
            "one nearer the camera), so the pitch ends were swapped to the "
            "canonical camera-on-y<0 frame"
        )
    else:
        reason = "already correctly oriented (near touchline at y=0)"
    return CalibrationCheck(H_canon / np.linalg.norm(H_canon), True, flipped, reason)
