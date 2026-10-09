"""End-to-end video pipeline on a synthetic clip (no GPU, no network).

A panning/zooming synthetic broadcast clip is rendered with players moving
at known pitch positions in two kit colours; their exact boxes are written
into the detection cache, so ``run_video`` runs fully offline with
``--detector none``.  The exported tracks are then checked against ground
truth: positions in metres, stable identities, consistent teams, and the
CSV schema.

Run:  .venv/bin/python -m pytest -q tests/test_video_pipeline.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict

import cv2
import numpy as np
import pytest

from detection import Detection
from pitchcal.geometry import apply_h_w
from pitchcal.synthetic import render_frame
from video.export import TRACK_COLUMNS
from video.run_video import run

N_FRAMES = 36
FPS = 25.0
SIZE = (1280, 720)
KITS = {0: (40, 40, 200), 1: (210, 150, 40)}  # BGR: red, blue


def _players(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10 players near the camera's path: start xy, velocity (m/s), team."""
    start = np.column_stack([rng.uniform(38, 58, 10), rng.uniform(22, 46, 10)])
    vel = rng.normal(0, 2.5, (10, 2))
    team = np.array([0, 1] * 5)
    return start, vel, team


def _make_clip(path: str, cache_path: str) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(5)
    start, vel, team = _players(rng)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    cache, gt = {}, {}
    for i in range(N_FRAMES):
        a = i / (N_FRAMES - 1)
        fr = render_frame(np.array([52.5, -35.0, 22.0]), np.array([42 + 12 * a, 32.0, 0.0]),
                          1550.0 + 200 * a, image_size=SIZE, n_players=0,
                          rng=np.random.default_rng(100 + i))
        img = fr.image
        pos = start + vel * (i / FPS)
        gt[i] = pos
        dets = []
        for p, t in zip(pos, team):
            foot, _ = apply_h_w(fr.H_gt, p[None])
            head, _ = apply_h_w(fr.H_gt, (p + [0, 0.6])[None])
            px_per_m = float(np.linalg.norm(head[0] - foot[0])) / 0.6
            h = 1.8 * px_per_m
            w = h / 3
            fx, fy = foot[0]
            cv2.rectangle(img, (int(fx - w / 2), int(fy - h)), (int(fx + w / 2), int(fy)), KITS[t], -1)
            cv2.rectangle(img, (int(fx - w / 2), int(fy - 0.3 * h)), (int(fx + w / 2), int(fy)),
                          (30, 30, 30), -1)  # shorts/legs
            dets.append(Detection("player", 0.9, (fx, fy - h / 2), (w, h), (fx, fy)))
        cache[str(i)] = [asdict(d) for d in dets]
        writer.write(img)
    writer.release()
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return {"gt": gt, "team": team}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("video")
    clip = str(d / "clip.mp4")
    out = str(d / "out")
    os.makedirs(out)
    truth = _make_clip(clip, os.path.join(out, "detections.json"))
    summary = run(clip, out, detector="none", detect_stride=2, render=True)
    with open(os.path.join(out, "tracks.csv")) as f:
        rows = list(csv.DictReader(f))
    return summary, rows, truth, out


def test_camera_calibrated_throughout(result) -> None:
    summary, *_ = result
    assert summary["frames"] == N_FRAMES
    assert summary["calibrated_fraction"] == 1.0, summary["camera_sources"]


def test_positions_identities_and_teams(result) -> None:
    _, rows, truth, _ = result
    gt, team = truth["gt"], truth["team"]
    by_track = defaultdict(list)
    for r in rows:
        by_track[int(r["track_id"])].append(r)
    errs, owner = [], {}
    for tid, rs in by_track.items():
        hits = []
        for r in rs:
            p = np.array([float(r["x_m"]), float(r["y_m"])])
            d = np.linalg.norm(gt[int(r["frame"])] - p, axis=1)
            hits.append(int(d.argmin()))
            errs.append(float(d.min()))
        # each track follows ONE ground-truth player (no identity swaps)
        assert len(set(hits)) == 1, (tid, hits)
        owner[tid] = hits[0]
    assert np.median(errs) < 0.5 and np.percentile(errs, 95) < 1.2, np.percentile(errs, [50, 95])
    # every player tracked, and by a single track (fragments stitched)
    assert sorted(owner.values()) == list(range(10)), owner
    # team labels are consistent with the kits (up to label permutation)
    pairs = {(int(by_track[t][0]["team"]), int(team[g])) for t, g in owner.items()}
    assert len(pairs) == 2 and len({a for a, _ in pairs}) == 2, pairs


def test_outputs_and_schema(result) -> None:
    _, rows, _, out = result
    assert list(rows[0].keys()) == TRACK_COLUMNS
    for name in ("tracks.csv", "camera.csv", "summary.json", "minimap.mp4"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    cap = cv2.VideoCapture(os.path.join(out, "minimap.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
    cap.release()
    xs = [float(r["x_m"]) for r in rows]
    ys = [float(r["y_m"]) for r in rows]
    assert -4 <= min(xs) and max(xs) <= 109 and -4 <= min(ys) and max(ys) <= 72
