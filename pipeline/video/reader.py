"""reader.py -- sequential video frame access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int


def video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str, start: int = 0, max_frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, bgr)`` for every frame from ``start`` onwards.

    Every frame is decoded (the camera tracker needs consecutive frames);
    detection stride is applied by the caller, not here.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while max_frames is None or idx - start < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()
