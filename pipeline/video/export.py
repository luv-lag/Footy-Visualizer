"""export.py -- write the video pipeline's tables to CSV.

``tracks.csv``: one row per (frame, track), see :class:`positions.PositionRow`.
``camera.csv``: one row per frame -- camera source, fit, and the 3x3
pitch->undistorted-image homography (row-major), so positions can be
re-derived or audited without re-running the pipeline.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, fields

import numpy as np

from video.camera import CameraState
from video.positions import PositionRow

TRACK_COLUMNS = [f.name for f in fields(PositionRow)]
CAMERA_COLUMNS = ["frame", "source", "fit", "explained", "lam"] + [
    f"h{r}{c}" for r in range(3) for c in range(3)
]


def write_tracks_csv(rows: list[PositionRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACK_COLUMNS)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            for k in ("x_m", "y_m", "speed_mps", "conf", "t_s"):
                d[k] = "" if not np.isfinite(d[k]) else round(d[k], 3)
            w.writerow(d)


def write_camera_csv(states: list[CameraState], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CAMERA_COLUMNS)
        for s in states:
            H = s.H.ravel().tolist() if s.ok else [""] * 9
            fin = lambda v: "" if not np.isfinite(v) else round(float(v), 4)  # noqa: E731
            w.writerow([s.frame_idx, s.source, fin(s.fit), fin(s.explained),
                        "" if s.dm is None else s.dm.lam, *H])
