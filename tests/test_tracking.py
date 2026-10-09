"""Pitch-space tracking and offline stitching (pure logic, fast)."""

from __future__ import annotations

import numpy as np

from detection import Detection
from pitchcal import DivisionModel
from pitchcal.synthetic import camera_homography
from video.camera import CameraState
from video.tracking import PlayerTracker, TrackSummary, stitch_tracklets


def _summary(tid, f0, f1, xy0, xy1, team=0, cls="player"):
    return TrackSummary(tid, f0, f1, np.array(xy0, float), np.array(xy1, float), team, cls)


def test_stitch_joins_reachable_fragment_only() -> None:
    a = _summary(1, 0, 50, (10, 10), (20, 10))
    b = _summary(2, 75, 120, (23, 11), (30, 12))  # 1 s later, 3 m away: same player
    c = _summary(3, 60, 90, (60, 40), (61, 40))  # far away: someone else
    d = _summary(4, 80, 99, (21, 10), (22, 10), team=1)  # reachable but other team
    m = stitch_tracklets([a, b, c, d], fps=25.0)
    assert m[2] == 1 and m[3] == 3 and m[4] == 4


def test_pitch_space_ids_survive_a_fast_pan_and_zoom() -> None:
    """Static players, camera panning 1 m and zooming per frame: ids must hold."""
    size = (1280, 720)
    players = np.array([[40.0, 30.0], [42.0, 31.0], [50.0, 25.0], [55.0, 40.0]])
    tracker = PlayerTracker(kind="bytetrack", fps=25.0)
    seen: dict[int, set] = {}
    for i in range(30):
        H = camera_homography(np.array([52.5, -35, 22]), np.array([40 + i, 32, 0]), 1500 + 20 * i, size)
        state = CameraState(i, H, DivisionModel(0.0, size), "test")
        feet = state.pitch_to_image(players)
        dets = [Detection("player", 0.9, (u, v - 20), (12, 40), (u, v)) for u, v in feet]
        for td in tracker.update(dets, state, i):
            k = int(np.argmin(np.linalg.norm(feet - np.array(td.det.foot_point), axis=1)))
            seen.setdefault(td.track_id, set()).add(k)
    assert len(seen) == len(players) and all(len(v) == 1 for v in seen.values()), seen
