"""run_video.py -- video in, minimap video + track/camera CSVs out.

Phase 1 of the video pipeline: one continuous camera (a wide / tactical
cam, or a single broadcast shot).  Hard cuts are not detected yet -- the
camera tracker just reports itself lost across them and re-anchors.

Usage (project venv; add ``--anchor pitchcal+pnlcalib`` in the PnLCalib
venv for the GPU fallback):

    .venv/bin/python pipeline/video/run_video.py clip.mp4 --out out/clip \\
        --detector hosted --detect-stride 3

Outputs in ``--out``:
    tracks.csv      per (frame, track) pitch positions (see export.py)
    camera.csv      per-frame camera source / fit / homography
    minimap.mp4     annotated frame | top-down minimap, with a status strip
    summary.json    run statistics (calibrated fraction, gates, timings)
    detections.json detection cache (re-runs skip detection entirely)

Two passes over the file: a forward pass (detect, follow the camera, take
jersey colour signatures), then an offline pass -- smooth the camera, track
players in *pitch space* through the smoothed camera, vote teams per
track, stitch and smooth tracks -- before rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../pipeline/video
_PIPELINE_DIR = os.path.dirname(_HERE)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from video.camera import (  # noqa: E402
    CameraState,
    CameraTracker,
    pitchcal_anchor,
    pitchcal_then_pnlcalib_anchor,
    smooth_states,
)
from video.detectors import CachedDetector, make_detector  # noqa: E402
from video.export import write_camera_csv, write_tracks_csv  # noqa: E402
from video.minimap import Minimap, annotate_frame, compose  # noqa: E402
from video.positions import build_rows  # noqa: E402
from video.reader import iter_frames, video_info  # noqa: E402
from video.teams import TeamVoter  # noqa: E402
from video.tracking import PERSON_CLASSES, PlayerTracker, detections_to_xyxy  # noqa: E402


def run(
    video: str,
    out_dir: str,
    detector: str = "hosted",
    detect_stride: int = 3,
    tracker: str = "bytetrack",
    anchor: str = "pitchcal",
    device: str = "cuda:0",
    refine_every: int = 5,
    start: int = 0,
    max_frames: int | None = None,
    render: bool = True,
) -> dict:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    info = video_info(video)
    fps = info.fps
    stride = max(1, detect_stride)

    det = CachedDetector(make_detector(detector), os.path.join(out_dir, "detections.json"))
    anchor_fn = pitchcal_anchor if anchor == "pitchcal" else pitchcal_then_pnlcalib_anchor(device)
    camera = CameraTracker(anchor_fn=anchor_fn, refine_every=refine_every)
    players = PlayerTracker(kind=tracker, fps=fps, stride=stride)
    voter = TeamVoter()

    # ---- pass 1: forward -------------------------------------------------
    states: dict[int, CameraState] = {}
    detections: dict[int, list] = {}  # detection frame -> detections
    signatures: dict[int, list] = {}  # detection frame -> per-detection signature
    balls = {}
    boxes = feet = None
    timings: Counter = Counter()
    n = 0
    for idx, frame in iter_frames(video, start=start, max_frames=max_frames):
        if (idx - start) % stride == 0:
            t0 = time.time()
            dets = det.detect(frame, idx)
            timings["detect_s"] += time.time() - t0
            t0 = time.time()
            persons = [d for d in dets if d.class_name in PERSON_CLASSES]
            boxes = detections_to_xyxy(persons)
            feet = np.array([d.foot_point for d in persons]).reshape(-1, 2)
            detections[idx] = dets
            signatures[idx] = [TeamVoter.signature(frame, d) for d in dets]
            ball = [d for d in dets if d.class_name == "ball"]
            if ball:
                balls[idx] = max(ball, key=lambda d: d.confidence)
            timings["signature_s"] += time.time() - t0
        t0 = time.time()
        states[idx] = camera.update(frame, idx, player_boxes=boxes, player_feet=feet)
        for b in camera.pop_backfilled():
            states[b.frame_idx] = b
        timings["camera_s"] += time.time() - t0
        n += 1
        if n % 100 == 0:
            ok = sum(s.ok for s in states.values()) / len(states)
            print(f"[pass 1] frame {idx}  calibrated {ok:.0%}", flush=True)
            det.save()
    det.save()
    if not states:
        raise RuntimeError(f"no frames decoded from {video}")

    # ---- pass 2: offline -------------------------------------------------
    t0 = time.time()
    ordered = smooth_states([states[i] for i in sorted(states)])
    states = {s.frame_idx: s for s in ordered}
    tracked = []
    for idx in sorted(detections):
        dets = detections[idx]
        sig_of = {id(d): sig for d, sig in zip(dets, signatures[idx])}
        for td in players.update(dets, states.get(idx), idx):
            tracked.append(td)
            voter.add(td.track_id, sig_of[id(td.det)])
    teams = voter.assign()
    rows, gate_stats = build_rows(tracked, balls, states, teams, fps, stride=stride)
    write_tracks_csv(rows, os.path.join(out_dir, "tracks.csv"))
    write_camera_csv(ordered, os.path.join(out_dir, "camera.csv"))
    timings["offline_s"] += time.time() - t0

    # ---- pass 3: render --------------------------------------------------
    if render:
        t0 = time.time()
        _render(video, out_dir, rows, states, fps, start, max_frames)
        timings["render_s"] += time.time() - t0

    sources = Counter(s.source.split(":")[0] for s in ordered)
    summary = {
        "video": os.path.abspath(video),
        "frames": len(ordered),
        "fps": fps,
        "detect_stride": stride,
        "detector": detector,
        "tracker": tracker,
        "anchor": anchor,
        "calibrated_fraction": round(sum(s.ok for s in ordered) / len(ordered), 4),
        "camera_sources": dict(sources),
        "team_separation_score": voter.separation_score,
        "gates": gate_stats,
        "rows": len(rows),
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "wall_s": round(time.time() - t_start, 2),
        "frames_per_s": round(len(ordered) / max(time.time() - t_start, 1e-9), 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _render(video, out_dir, rows, states, fps, start, max_frames) -> None:
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r.frame].append(r)
    mm = Minimap()
    writer = None
    try:
        for idx, frame in iter_frames(video, start=start, max_frames=max_frames):
            st = states.get(idx)
            fr_rows = by_frame.get(idx, [])
            n_people = sum(r.cls != "ball" for r in fr_rows)
            src = st.source if st is not None else "none"
            status = f"t={idx / fps:6.1f}s   camera: {src}   tracked: {n_people}"
            img = compose(annotate_frame(frame, fr_rows, st), mm.render(fr_rows), status,
                          ok=st is not None and st.ok)
            if writer is None:
                path = os.path.join(out_dir, "minimap.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--detector", default="hosted", choices=["hosted", "local", "none"],
                   help="'none' = use the detection cache only")
    p.add_argument("--detect-stride", type=int, default=3,
                   help="detect every N-th frame (positions are interpolated between)")
    p.add_argument("--tracker", default="bytetrack",
                   choices=["bytetrack", "mcbyte", "ocsort", "sort", "botsort"],
                   help="association algorithm (runs in pitch space)")
    p.add_argument("--anchor", default="pitchcal", choices=["pitchcal", "pitchcal+pnlcalib"])
    p.add_argument("--device", default="cuda:0", help="PnLCalib device for --anchor pitchcal+pnlcalib")
    p.add_argument("--refine-every", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args(argv)
    summary = run(a.video, a.out, a.detector, a.detect_stride, a.tracker, a.anchor, a.device,
                  a.refine_every, a.start, a.max_frames, not a.no_render)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
