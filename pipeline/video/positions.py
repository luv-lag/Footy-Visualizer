"""positions.py -- tracked detections + camera states -> per-frame pitch rows.

The offline "global" pass of the video pipeline:

1. project every tracked foot point through its frame's camera state;
2. drop detections that land off the pitch (stands, scoreboard graphics,
   ball boys) -- a plausibility gate the single-frame pipeline lacked;
3. stitch track fragments in pitch space (:func:`tracking.stitch_tracklets`);
4. per track, interpolate across detection stride / short dropouts and
   smooth in pitch space (Savitzky-Golay);
5. flag physically implausible speeds instead of silently drawing them.

All positions are corner-origin metres, (0, 0)-(105, 68), as in
``frame_analysis``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from detection import Detection
from video.camera import CameraState
from video.tracking import TrackedDetection, TrackSummary, stitch_tracklets

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass
class PositionRow:
    """One (frame, track) sample -- the unit of the CSV export."""

    frame: int
    t_s: float
    track_id: int  # -1 for the ball
    cls: str
    team: int  # -1 = unknown / not an outfield player
    x_m: float
    y_m: float
    speed_mps: float
    kind: str  # "detected" | "interpolated"
    conf: float  # detector confidence (NaN when interpolated)
    camera: str  # camera-state source for this frame
    flags: str  # ";"-joined, e.g. "speed_outlier"


def _on_pitch(xy: np.ndarray, margin: float) -> bool:
    return bool(
        -margin <= xy[0] <= PITCH_LENGTH_M + margin
        and -margin <= xy[1] <= PITCH_WIDTH_M + margin
    )


def build_rows(
    tracked: list[TrackedDetection],
    balls: dict[int, Detection],
    states: dict[int, CameraState],
    teams: dict[int, tuple[int, float]],
    fps: float,
    stride: int = 1,
    off_pitch_margin_m: float = 2.0,
    max_interp_gap_s: float = 1.0,
    smooth_s: float = 0.4,
    max_speed_mps: float = 12.0,
    stitch: bool = True,
) -> tuple[list[PositionRow], dict]:
    """Assemble the per-frame position table.

    Args:
        tracked: confirmed tracked detections from every detection frame.
        balls: best ball detection per detection frame.
        states: camera state per frame index (every decoded frame).
        teams: ``{track_id: (team, vote_fraction)}`` from :class:`TeamVoter`.
        fps: source frame rate.
        stride: detection stride used for ``tracked``.

    Returns:
        (rows, stats) -- rows sorted by (frame, track_id); stats counts
        what the gates removed, for the run summary.
    """
    stats: Counter = Counter()

    # 1-2. project + off-pitch gate
    obs: dict[int, list[tuple[int, np.ndarray, Detection]]] = defaultdict(list)
    for td in tracked:
        st = states.get(td.frame_idx)
        if st is None or not st.ok:
            stats["no_camera"] += 1
            continue
        xy = st.image_to_pitch(np.array([td.det.foot_point]))[0]
        if not np.all(np.isfinite(xy)) or not _on_pitch(xy, off_pitch_margin_m):
            stats["off_pitch"] += 1
            continue
        obs[td.track_id].append((td.frame_idx, xy, td.det))
    stats["tracks_raw"] = len(obs)

    def cls_of(items) -> str:
        return Counter(d.class_name for _, _, d in items).most_common(1)[0][0]

    def team_of(tid: int, cls: str) -> int:
        return teams[tid][0] if cls == "player" and tid in teams else -1

    # 3. stitch fragments in pitch space (identity mapping when disabled)
    for items in obs.values():
        items.sort(key=lambda o: o[0])
    summaries = [
        TrackSummary(tid, items[0][0], items[-1][0], items[0][1], items[-1][1],
                     team_of(tid, cls_of(items)), cls_of(items))
        for tid, items in obs.items()
    ]
    mapping = stitch_tracklets(summaries, fps) if stitch else {s.track_id: s.track_id for s in summaries}
    merged: dict[int, list] = defaultdict(list)
    member_teams: dict[int, list[int]] = defaultdict(list)
    for summ in summaries:
        merged[mapping[summ.track_id]].extend(obs[summ.track_id])
        if summ.team >= 0:
            member_teams[mapping[summ.track_id]].append(summ.team)
    obs = merged
    final_team = {tid: Counter(ts).most_common(1)[0][0] for tid, ts in member_teams.items()}
    stats["tracks_stitched"] = len(obs)

    # 4-5. interpolate, smooth, speed-gate
    rows: list[PositionRow] = []
    max_gap = max(stride, int(round(max_interp_gap_s * fps)))
    win = max(5, int(round(smooth_s * fps)) | 1)
    for tid, items in obs.items():
        items.sort(key=lambda o: o[0])
        c = cls_of(items)
        team = final_team.get(tid, -1) if c == "player" else -1
        frames = np.array([f for f, _, _ in items])
        xy = np.stack([p for _, p, _ in items])
        conf = {f: d.confidence for f, _, d in items}
        for seg in _segments(frames, max_gap):
            f_seg = frames[seg]
            dense = np.arange(f_seg[0], f_seg[-1] + 1)
            x = np.interp(dense, f_seg, xy[seg, 0])
            y = np.interp(dense, f_seg, xy[seg, 1])
            if len(dense) >= win:
                x = savgol_filter(x, win, 2)
                y = savgol_filter(y, win, 2)
            speed = np.zeros(len(dense))
            if len(dense) > 1:
                speed = np.hypot(np.gradient(x), np.gradient(y)) * fps
            detected = set(f_seg.tolist())
            for k, f in enumerate(dense):
                st = states.get(int(f))
                if f not in detected and (st is None or not st.ok):
                    continue  # never interpolate through frames with no pitch view
                flags = []
                if speed[k] > max_speed_mps:
                    flags.append("speed_outlier")
                    stats["speed_outlier"] += 1
                rows.append(PositionRow(
                    frame=int(f), t_s=f / fps, track_id=int(tid), cls=c, team=team,
                    x_m=float(x[k]), y_m=float(y[k]), speed_mps=float(speed[k]),
                    kind="detected" if f in detected else "interpolated",
                    conf=float(conf.get(int(f), float("nan"))),
                    camera=st.source if st is not None else "none",
                    flags=";".join(flags),
                ))

    # the ball: projected as-is (a ball in the air projects wrongly onto the
    # ground plane, so it is flagged rather than smoothed into a track)
    for f, det in balls.items():
        st = states.get(f)
        if st is None or not st.ok:
            continue
        xy = st.image_to_pitch(np.array([det.foot_point]))[0]
        if not _on_pitch(xy, off_pitch_margin_m):
            continue
        rows.append(PositionRow(int(f), f / fps, -1, "ball", -1, float(xy[0]), float(xy[1]),
                                float("nan"), "detected", det.confidence, st.source,
                                "ground_plane_assumed"))

    # Renumber tracks 1..N in order of first appearance: raw ids carry the
    # tracker's / colour-splitter's internal offsets and are not meant for
    # display.  (-1 stays the ball.)
    first: dict[int, int] = {}
    for r in sorted(rows, key=lambda r: r.frame):
        if r.track_id >= 0 and r.track_id not in first:
            first[r.track_id] = len(first) + 1
    for r in rows:
        if r.track_id >= 0:
            r.track_id = first[r.track_id]
    rows.sort(key=lambda r: (r.frame, r.track_id))
    return rows, dict(stats)


def _segments(frames: np.ndarray, max_gap: int) -> list[slice]:
    """Split sorted frame indices wherever the gap exceeds ``max_gap``."""
    breaks = np.where(np.diff(frames) > max_gap)[0] + 1
    bounds = [0, *breaks.tolist(), len(frames)]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
