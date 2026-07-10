"""pitchcal: single-frame broadcast-to-pitch homography calibration.

Maps pixel coordinates in one static football broadcast frame to metric
pitch coordinates using the pitch's own markings (lines, centre circle,
penalty arcs), with explicit handling of the pitch-symmetry orientation
ambiguity and per-frame failure reporting.

Public entry points:

* :func:`calibrate` -- image (+ optional player positions) -> result.
* :class:`CalibrationResult` -- mapping + status + confidence.
* :class:`CalibrationStatus` -- OK / INSUFFICIENT_FEATURES / ...
* :class:`PitchModel` -- template geometry (override for non-FIFA sizes).
* :func:`render_overlay` -- draw the calibrated template onto a frame.

CPU-only: no GPU memory is used, so it can run alongside detection /
team-ID stages without sharing the VRAM budget.
"""

from .api import CalibrationResult, CalibrationStatus, calibrate, render_overlay
from .orientation import CalibrationCheck, validate_and_canonicalize_homography
from .geometry import DivisionModel
from .pitch_model import PitchModel

__all__ = [
    "CalibrationCheck",
    "CalibrationResult",
    "CalibrationStatus",
    "DivisionModel",
    "PitchModel",
    "calibrate",
    "render_overlay",
    "validate_and_canonicalize_homography",
]
