"""
io_utils.py — shared I/O helpers for approach_4 modules.

Nothing here is pipeline logic; only encoding, file read/write, and JSON
serialisation of numpy types so the rest of the code stays clean.
"""
from __future__ import annotations
import base64
import json
import os

import cv2
import numpy as np


# ── image encoding ────────────────────────────────────────────────────────────

def encode_image_b64(image_bgr: np.ndarray) -> str:
    """JPEG-encode a BGR image and return as a base64 string (for Roboflow API).

    Lifted from pitch_analysis/football_analysis.py and pitch_homography.py
    (both had identical private copies).
    """
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── JSON helpers ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── calibration result serialisation ─────────────────────────────────────────

def calib_result_to_dict(result) -> dict:
    """Serialise a CalibResult to a JSON-safe dict.

    Field layout matches PnLCalib/extract_transform.py output schema so the
    two tools produce interchangeable artefacts.
    """
    if result.grade.value == "abstain":
        return {
            "grade":               result.grade.value,
            "n_kp":                result.n_kp,
            "reprojection_error_px": result.reproj_err,
            "mode":                result.mode,
            "H":                   None,
            "K":                   None,
            "R":                   None,
            "t":                   None,
        }
    return {
        "grade":                     result.grade.value,
        "n_kp":                      result.n_kp,
        "reprojection_error_px":     result.reproj_err,
        "mode":                      result.mode,
        "pan_deg":                   result.pan_deg,
        "tilt_deg":                  result.tilt_deg,
        "roll_deg":                  result.roll_deg,
        "intrinsics_K": {
            "fx": float(result.K[0, 0]),
            "fy": float(result.K[1, 1]),
            "cx": float(result.K[0, 2]),
            "cy": float(result.K[1, 2]),
            "matrix": result.K.tolist(),
        },
        "rotation_R":                result.R.tolist(),
        "camera_position_world_m":   result.cam_pos.tolist(),
        "translation_t_cam":         result.t.tolist(),
        "homography_H_world_to_image": result.H.tolist(),
        # Coordinate system — canonical description from extract_transform.py
        "coord_system": (
            "World origin = pitch centre. "
            "X: length axis ±52.5 m, Y: width axis ±34 m, Z: up. "
            "H maps ground-plane world [Xw, Yw, 1] -> image pixel [u/w, v/w]."
        ),
    }
