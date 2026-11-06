"""minimap.py -- fast per-frame rendering for the minimap video.

``tactical_map.py`` (matplotlib) is right for stills but far too slow per
frame, so this renders directly with OpenCV: the pitch template is drawn
once into a background image, and each frame only stamps markers on a
copy of it.  Colors come from ``drawing.py`` so stills and video match.
"""

from __future__ import annotations

import cv2
import numpy as np

from drawing import CLASS_COLORS, TEAM_COLORS, scale_factor
from pitchcal import PitchModel
from video.camera import CameraState
from video.positions import PositionRow

GRASS = (58, 125, 45)  # BGR
LINE = (235, 235, 235)
NO_TEAM = (170, 170, 170)
STRIP_OK = (40, 90, 30)
STRIP_BAD = (30, 30, 150)


def marker_color(row: PositionRow) -> tuple[int, int, int]:
    if row.cls == "player":
        return TEAM_COLORS.get(row.team, NO_TEAM)
    return CLASS_COLORS.get(row.cls, NO_TEAM)


class Minimap:
    """Top-down pitch canvas (corner-origin metres, near touchline at bottom)."""

    def __init__(self, px_per_m: float = 8.0, margin_m: float = 4.0,
                 pitch: PitchModel | None = None) -> None:
        self.pitch = pitch or PitchModel()
        self.s = px_per_m
        self.m = margin_m
        self.w = int(round((self.pitch.length + 2 * margin_m) * px_per_m))
        self.h = int(round((self.pitch.width + 2 * margin_m) * px_per_m))
        self.background = np.full((self.h, self.w, 3), GRASS, np.uint8)
        for _, pts in self.pitch.sample_all(step=0.25):
            # A template curve can have several disjoint pieces (e.g. the
            # penalty-area sides at both ends); split where samples jump.
            breaks = np.where(np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1.0)[0] + 1
            for piece in np.split(pts, breaks):
                if len(piece) > 1:
                    cv2.polylines(self.background, [self.to_px(piece)], False, LINE, 2, cv2.LINE_AA)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """Pitch metres (N, 2) -> canvas pixels; y is flipped so y=0 is at the bottom."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        u = (xy[:, 0] + self.m) * self.s
        v = (self.pitch.width + self.m - xy[:, 1]) * self.s
        return np.round(np.stack([u, v], axis=1)).astype(np.int32)

    def render(self, rows: list[PositionRow]) -> np.ndarray:
        img = self.background.copy()
        r = max(4, int(round(0.9 * self.s)))
        for row in rows:
            if "speed_outlier" in row.flags:
                continue
            (u, v), = self.to_px(np.array([[row.x_m, row.y_m]]))
            if row.cls == "ball":
                cv2.circle(img, (int(u), int(v)), max(3, r // 2), CLASS_COLORS["ball"], -1, cv2.LINE_AA)
                continue
            cv2.circle(img, (int(u), int(v)), r, marker_color(row), -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), r, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(img, str(row.track_id), (int(u) + r, int(v) - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def annotate_frame(frame: np.ndarray, rows: list[PositionRow], state: CameraState | None) -> np.ndarray:
    """Draw each tracked person's foot position back onto the broadcast frame.

    Markers are re-projected from the (smoothed) pitch positions through the
    frame's camera, so interpolated frames get markers too, and what is
    drawn on the frame is exactly what the minimap claims.
    """
    out = frame.copy()
    if state is None or not state.ok:
        return out
    k = scale_factor(frame)
    people = [r for r in rows if r.cls != "ball" and "speed_outlier" not in r.flags]
    if not people:
        return out
    px = state.pitch_to_image(np.array([[r.x_m, r.y_m] for r in people]))
    for row, (u, v) in zip(people, px):
        if not (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
            continue
        axes = (int(14 * k), int(5 * k))
        cv2.ellipse(out, (int(u), int(v)), axes, 0, 0, 360, marker_color(row),
                    max(1, int(2 * k)), cv2.LINE_AA)
        cv2.putText(out, str(row.track_id), (int(u) - axes[0], int(v) + 3 * axes[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, (255, 255, 255), max(1, int(k)), cv2.LINE_AA)
    return out


def compose(frame_annot: np.ndarray, minimap: np.ndarray, status: str, ok: bool,
            out_height: int = 540) -> np.ndarray:
    """Side-by-side frame | minimap with a status strip on top."""
    def fit_h(img):
        s = out_height / img.shape[0]
        return cv2.resize(img, (int(round(img.shape[1] * s)), out_height), interpolation=cv2.INTER_AREA)

    body = np.hstack([fit_h(frame_annot), fit_h(minimap)])
    strip = np.full((32, body.shape[1], 3), STRIP_OK if ok else STRIP_BAD, np.uint8)
    cv2.putText(strip, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([strip, body])
    # Video codecs want even dimensions.
    return out[: out.shape[0] // 2 * 2, : out.shape[1] // 2 * 2]
