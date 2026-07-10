"""frame_analysis.py -- coordinate-frame conversion between PnLCalib and
pitchcal's canonical pitch frame.

pitchcal's canonical frame is corner-origin: (0, 0) to (105, 68) metres.
PnLCalib's production wrapper works in a CENTRE-origin world frame instead.
Before any PnLCalib-derived position or homography can be compared with, or
fed into, anything downstream, it has to be converted into the corner frame
-- and that conversion is not just a translation.
"""

from __future__ import annotations

import numpy as np

# Pitch dimensions shared by both systems' fitted models (FIFA standard).
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0


# ──────────────────────────────────────────────────────────────────────────
# Coordinate conversion (PnLCalib fallback path only)
# ──────────────────────────────────────────────────────────────────────────
def center_to_corner(
    x_center: float,
    y_center: float,
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
) -> tuple[float, float]:
    """Convert PnLCalib centre-origin world metres to corner-origin metres.

    The validated conversion is  **corner (x, y) = (x_c + L/2, W/2 - y_c)**
    (equivalently: centre (x, y) = (x - L/2, W/2 - y)); for a 105 x 68 pitch
    that is ``(x_c + 52.5, 34 - y_c)``.

    THE Y SIGN FLIP IS ESSENTIAL: PnLCalib's world +Y points the OPPOSITE
    way to the corner frame's +Y.  A naive translation-only conversion
    (``x_c + 52.5, y_c + 34``) silently mirrors every position across the
    pitch's long axis -- and the bug is invisible in any overlay/top-down
    visual check, because the pitch marking layout is symmetric under that
    exact flip.  It only shows up in actual point positions (and in the
    handedness test).  This was discovered and fixed in
    ``eval/eval_comparison/run_comparison.py`` (see ``T_CENTER_TO_CORNER`` there
    and the derivation in ``eval/eval_comparison/validate_orientation_check.py``);
    do not "simplify" this function back to a translation.

    Args:
        x_center: PnLCalib world X (length axis, metres, pitch centre = 0).
        y_center: PnLCalib world Y (width axis, metres, pitch centre = 0).
        length: pitch length in metres (default FIFA 105).
        width: pitch width in metres (default FIFA 68).

    Returns:
        (x, y) in the corner-origin frame: x in [0, length], y in [0, width]
        for on-pitch points.
    """
    return x_center + length / 2.0, width / 2.0 - y_center


# Homogeneous point-map matrices for the SAME validated conversion as
# :func:`center_to_corner`, used to change a homography's pitch-coordinate
# frame (right-composition): H_corner = H_center_raw @ _CORNER_TO_CENTER_XY
# and H_center = H_corner @ _CENTER_TO_CORNER_XY.  Both embed the essential
# Y sign flip -- see center_to_corner's docstring; precedent:
# eval/eval_comparison/run_comparison.py (T_CENTER_TO_CORNER).
_CORNER_TO_CENTER_XY: np.ndarray = np.array(
    [[1.0, 0.0, -PITCH_LENGTH_M / 2.0], [0.0, -1.0, PITCH_WIDTH_M / 2.0], [0.0, 0.0, 1.0]]
)
_CENTER_TO_CORNER_XY: np.ndarray = np.array(
    [[1.0, 0.0, PITCH_LENGTH_M / 2.0], [0.0, -1.0, PITCH_WIDTH_M / 2.0], [0.0, 0.0, 1.0]]
)
