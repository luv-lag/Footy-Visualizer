"""
projection.py — NEW connector: maps detected players to metric pitch coordinates.

This module bridges player detections (pipeline/detection.py) to PnLCalib
world coordinates (from calibration.py) -- there is no upstream equivalent;
compute_player_errors() in the original eval script used GT annotation foot
pixels, not detector bboxes.

World coordinate frame (same as calibration.py / PnLCalib):
  Origin : pitch centre.  X: length ±52.5 m.  Y: width ±34 m.
Out-of-bounds: |Xw| > 52.5 m OR |Yw| > 34.0 m.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


# ── Pitch limits (FIFA standard) ──────────────────────────────────────────────
PITCH_HALF_LEN = 52.5   # metres, X axis
PITCH_HALF_WID = 34.0   # metres, Y axis


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ProjectedPlayer:
    """A single detected player with metric position.

    pixel_uv  : (u, v) foot pixel in the image.
    world_xy  : (Xw, Yw) in PnLCalib world metres; None only if H_inv failed.
    team      : 0 or 1 (from KMeans HSV assignment) or -1 if unknown.
    conf      : YOLO detection confidence.
    oob       : True if |Xw| > PITCH_HALF_LEN or |Yw| > PITCH_HALF_WID.
    det_index : original detection index (for traceability).
    """
    pixel_uv  : tuple[float, float]
    world_xy  : tuple[float, float] | None
    team      : int
    conf      : float
    oob       : bool
    det_index : int


# ── Core math (lifted verbatim from evaluate_synloc_calibration.project_i2w()) ──

def _project_i2w(
    u: float, v: float, H_inv: np.ndarray
) -> tuple[float, float]:
    """Back-project image pixel (u, v) to PnLCalib world (Xw, Yw).

    Identical to evaluate_synloc_calibration.py lines 110–112.
    H_inv is the 3x3 inverse of the ground-plane homography H.
    """
    q = H_inv @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


def _foot_point(bbox) -> tuple[float, float]:
    """Return the foot-contact pixel (bottom-centre) of a bounding box.

    Convention: u = (x1+x2)/2, v = y2.

    Accepts:
      - list/tuple [x1, y1, x2, y2]  (player_detection.py format)
      - dict with keys 'x1'/'y1'/'x2'/'y2'
      - dict with keys 'xmin'/'ymin'/'xmax'/'ymax'
      - dict with keys 'x'/'y'/'width'/'height' (centre+size)
    """
    if isinstance(bbox, (list, tuple)):
        x1, _y1, x2, y2 = [float(v) for v in bbox[:4]]
    elif "x1" in bbox:
        x1, x2, y2 = float(bbox["x1"]), float(bbox["x2"]), float(bbox["y2"])
    elif "xmin" in bbox:
        x1, x2, y2 = float(bbox["xmin"]), float(bbox["xmax"]), float(bbox["ymax"])
    else:
        # centre+size format  (xc, yc, w, h)
        xc, yc = float(bbox["x"]), float(bbox["y"])
        w,  h  = float(bbox["width"]), float(bbox["height"])
        x1, x2, y2 = xc - w / 2, xc + w / 2, yc + h / 2
    u = (x1 + x2) / 2.0
    return u, y2


# ── Public API ────────────────────────────────────────────────────────────────

def project_players(
    players     : list[dict],
    team_labels : list[int],
    H           : np.ndarray,
) -> list[ProjectedPlayer]:
    """Project each detected player's foot pixel to metric pitch coordinates.

    Parameters
    ----------
    players     : list of detection dicts from player_detection.py
                  Each dict must contain 'bbox' and 'conf'.
    team_labels : parallel list of team indices (0 or 1) from KMeans HSV.
                  If shorter than players (or empty), missing entries get team=-1.
    H           : (3,3) ground-plane homography from calibration.py
                  (world [Xw,Yw,1] → image [u/w, v/w]).

    Returns
    -------
    List of ProjectedPlayer, one per input detection.
    """
    H_inv = np.linalg.inv(H)
    results: list[ProjectedPlayer] = []

    for i, det in enumerate(players):
        bbox  = det.get("bbox") or det  # player_detection dicts nest bbox inside 'bbox'
        conf  = float(det.get("conf", det.get("confidence", 0.0)))
        team  = int(team_labels[i]) if i < len(team_labels) else -1

        u, v  = _foot_point(bbox)

        try:
            xw, yw = _project_i2w(u, v, H_inv)
        except (ZeroDivisionError, np.linalg.LinAlgError):
            results.append(ProjectedPlayer(
                pixel_uv=(u, v), world_xy=None,
                team=team, conf=conf, oob=False, det_index=i,
            ))
            continue

        oob = abs(xw) > PITCH_HALF_LEN or abs(yw) > PITCH_HALF_WID
        results.append(ProjectedPlayer(
            pixel_uv=(u, v), world_xy=(xw, yw),
            team=team, conf=conf, oob=oob, det_index=i,
        ))

    return results
