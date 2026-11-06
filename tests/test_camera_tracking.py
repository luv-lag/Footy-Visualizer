"""Camera tracking through a synthetic broadcast pan/zoom (no GPU, no network).

pitchcal's synthetic renderer gives the exact homography for every frame, so
the tracker's per-frame mapping can be compared against ground truth in
metres -- including whether the pitch ends ever swap mid-shot (an
orientation flip would show up as tens of metres of error).

Run:  .venv/bin/python -m pytest -q tests/test_camera_tracking.py
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchcal.synthetic import render_frame
from video.camera import CameraTracker, smooth_states

N_FRAMES = 30
MAX_ERR_M = 0.6

_xs, _ys = np.meshgrid(np.linspace(100, 1180, 8), np.linspace(250, 700, 5))
_GRID = np.stack([_xs.ravel(), _ys.ravel()], axis=1)


def _pan_sequence(look_x0: float, look_x1: float) -> list:
    """Fixed tripod camera panning along the pitch while zooming in."""
    frames = []
    for i in range(N_FRAMES):
        a = i / (N_FRAMES - 1)
        frames.append(
            render_frame(
                np.array([52.5, -35.0, 22.0]),
                np.array([look_x0 + (look_x1 - look_x0) * a, 30.0 + 6.0 * a, 0.0]),
                1500.0 + 400.0 * a,
                n_players=0,
                rng=np.random.default_rng(i),  # fresh sensor noise per frame
            )
        )
    return frames


def _err_m(state, frame) -> float:
    """Median pitch-space error (m) of the tracked mapping on on-pitch pixels."""
    gt = frame.gt_image_to_pitch(_GRID)
    on = (gt[:, 0] > -5) & (gt[:, 0] < 110) & (gt[:, 1] > -5) & (gt[:, 1] < 73)
    est = state.image_to_pitch(_GRID)
    return float(np.median(np.linalg.norm(est[on] - gt[on], axis=1)))


@pytest.fixture(scope="module")
def late_anchor_sequence() -> list:
    # Starts on a view pitchcal refuses (too little paint), so the first
    # anchor only lands ~10 frames in -- exercising the backward fill too.
    return _pan_sequence(30.0, 70.0)


def test_tracks_pan_and_zoom_without_drift_or_flip(late_anchor_sequence) -> None:
    frames = late_anchor_sequence
    tracker = CameraTracker(refine_every=5)
    states = [tracker.update(f.image, i) for i, f in enumerate(frames)]
    for b in tracker.pop_backfilled():
        states[b.frame_idx] = b

    assert all(s.ok for s in states), [s.source for s in states]
    assert any(s.source == "backfill" for s in states)
    assert any(s.source.startswith("anchor") for s in states)
    errs = [_err_m(s, f) for s, f in zip(states, frames)]
    assert max(errs) < MAX_ERR_M, np.round(errs, 2)

    smoothed = smooth_states(states)
    errs_s = [_err_m(s, f) for s, f in zip(smoothed, frames)]
    assert max(errs_s) < MAX_ERR_M, np.round(errs_s, 2)


def test_non_pitch_frame_loses_track_honestly(late_anchor_sequence) -> None:
    frames = late_anchor_sequence
    tracker = CameraTracker(refine_every=5, max_backfill=0)
    states = [tracker.update(f.image, i) for i, f in enumerate(frames[:20])]
    assert states[-1].ok

    # A cut to a crowd shot: no grass, no flow -> lost, never a guessed H.
    crowd = np.random.default_rng(99).integers(40, 140, frames[0].image.shape, dtype=np.uint8)
    lost = tracker.update(crowd, 20)
    assert not lost.ok and lost.source == "lost"

    # Cutting back re-anchors.  (Cut to a later, paint-rich part of the pan:
    # the tracker inherits its anchor's single-frame accuracy, and pitchcal
    # alone is occasionally ~1 m off on individual frames of this pan.)
    tracker.reset()
    back = [tracker.update(f.image, 24 + i) for i, f in enumerate(frames[24:27])]
    assert back[0].ok and back[0].source.startswith("anchor")
    assert _err_m(back[-1], frames[26]) < MAX_ERR_M
