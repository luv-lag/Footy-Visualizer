"""tracking.py -- player identities across frames.

Online association runs in **pitch space**, not image space.  Broadcast
players are small (often 15-30 px wide) and the camera pans and zooms
constantly, so pixel-IoU trackers fragment badly even with camera-motion
compensation: on a synthetic pan with *perfect* detections, image-space
BoT-SORT produced 46 ids for 10 players.  Once the camera is calibrated,
each foot point is mapped to metres and represented by a fixed-size
pseudo-box there; any IoU tracker from Roboflow's ``trackers`` library
(Apache-2.0) then sees a static scene in which only the players move.

Offline, :func:`stitch_tracklets` joins remaining fragments (a player lost
behind another and re-acquired under a new id) using pitch-space
continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
import trackers

from detection import Detection
from video.camera import CameraState

#: Classes that get a track id (the ball is handled separately).
PERSON_CLASSES: tuple[str, ...] = ("player", "goalkeeper", "referee")


@dataclass
class TrackedDetection:
    """One detection with the id the online tracker gave it."""

    frame_idx: int
    track_id: int
    det: Detection

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        (cx, cy), (w, h) = self.det.bbox_center, self.det.bbox_size
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def detections_to_xyxy(dets: list[Detection]) -> np.ndarray:
    """(N, 4) x0, y0, x1, y1 boxes."""
    if not dets:
        return np.zeros((0, 4))
    c = np.array([d.bbox_center for d in dets], dtype=float)
    s = np.array([d.bbox_size for d in dets], dtype=float)
    return np.hstack([c - s / 2, c + s / 2])


class PlayerTracker:
    """Pitch-space multi-person tracker (call :meth:`reset` at shot cuts).

    Args:
        kind: "bytetrack" (default), "mcbyte" (McByte's clear-match locking,
            without its image-space mask cue), "ocsort", "sort" or "botsort"
            (without camera compensation, which pitch space makes redundant).
        fps: source video frame rate.
        stride: detection stride (the tracker only sees every stride-th
            frame, so its notion of frame rate is fps / stride).
        lost_seconds: how long a lost track may be re-acquired online.
        activation_conf: detection confidence needed to start a track.
            Lower than the library default: far-side players are small and
            the football detector's confidences there are often ~0.4-0.6.
        box_m: side of the pseudo-box around each foot point, in metres.
            It must exceed a sprinter's displacement between detections
            (9 m/s at 25 fps, stride 3 -> ~1.1 m) for IoU to stay non-zero,
            while staying small enough to separate players in contact.
    """

    #: pseudo-box scale: metres -> tracker units (keeps the trackers'
    #: pixel-tuned Kalman noise in a sensible range).
    UNITS_PER_M = 10.0

    def __init__(
        self,
        kind: str = "bytetrack",
        fps: float = 25.0,
        stride: int = 1,
        lost_seconds: float = 2.0,
        activation_conf: float = 0.4,
        box_m: float = 2.5,
    ) -> None:
        self.kind = kind
        self.rate = fps / max(1, stride)
        self.lost_frames = max(1, int(round(lost_seconds * self.rate)))
        self.activation_conf = activation_conf
        self.box_m = box_m
        self.reset()

    def reset(self) -> None:
        common = dict(lost_track_buffer=self.lost_frames, frame_rate=self.rate,
                      high_conf_det_threshold=max(0.5, self.activation_conf))
        if self.kind == "bytetrack":
            self._tracker = trackers.ByteTrackTracker(
                **common, track_activation_threshold=self.activation_conf)
        elif self.kind == "botsort":
            self._tracker = trackers.BoTSORTTracker(
                **common, track_activation_threshold=self.activation_conf, enable_cmc=False)
        elif self.kind == "mcbyte":
            self._tracker = trackers.McByteTracker(
                **common, track_activation_threshold=self.activation_conf,
                enable_cmc=False, enable_mask_manager=False)
        elif self.kind == "ocsort":
            self._tracker = trackers.OCSORTTracker(**common)
        elif self.kind == "sort":
            self._tracker = trackers.SORTTracker(
                lost_track_buffer=self.lost_frames, frame_rate=self.rate,
                track_activation_threshold=self.activation_conf)
        else:
            raise ValueError(f"unknown tracker {self.kind!r}")

    def update(
        self, dets: list[Detection], state: CameraState | None, frame_idx: int
    ) -> list[TrackedDetection]:
        """Associate one detection frame's people with existing tracks.

        Frames without a camera (lost / not yet anchored) only age the
        tracks.  Returns confirmed tracks only; unconfirmed detections
        (tracker id -1) are typically one-frame false positives.
        """
        persons = [d for d in dets if d.class_name in PERSON_CLASSES]
        if state is None or not state.ok or not persons:
            self._tracker.update(sv.Detections.empty())
            return []
        xy = state.image_to_pitch(np.array([d.foot_point for d in persons]))
        half = self.box_m / 2
        boxes = np.hstack([xy - half, xy + half]) * self.UNITS_PER_M
        sv_dets = sv.Detections(
            xyxy=boxes,
            confidence=np.array([d.confidence for d in persons], dtype=float),
            class_id=np.array([PERSON_CLASSES.index(d.class_name) for d in persons], dtype=int),
        )
        sv_dets.data["det_idx"] = np.arange(len(persons))
        out = self._tracker.update(sv_dets)
        if len(out) == 0:
            return []
        return [
            TrackedDetection(frame_idx, int(tid), persons[int(i)])
            for tid, i in zip(out.tracker_id, out.data["det_idx"])
            if tid >= 0
        ]


# ---------------------------------------------------------------------- #
# Offline stitching (pitch space)
# ---------------------------------------------------------------------- #
@dataclass
class TrackSummary:
    """What stitching needs to know about one track."""

    track_id: int
    first_frame: int
    last_frame: int
    first_xy: np.ndarray  # pitch metres at first_frame
    last_xy: np.ndarray  # pitch metres at last_frame
    team: int  # -1 = unknown
    cls: str
    members: list[int] = field(default_factory=list)


def stitch_tracklets(
    summaries: list[TrackSummary],
    fps: float,
    max_gap_s: float = 3.0,
    max_speed_mps: float = 8.0,
    slack_m: float = 2.0,
) -> dict[int, int]:
    """Merge fragments of the same player into one id.

    Fragment B may continue fragment A when B starts after A ends (within
    ``max_gap_s``), both have the same class and compatible team, and B's
    first position is reachable from A's last at a running speed of at
    most ``max_speed_mps`` (plus ``slack_m`` for projection noise).
    Candidate links are accepted greedily, cheapest (implied speed) first,
    with each fragment continuing at most one predecessor.

    Returns:
        ``{original_track_id: stitched_track_id}`` for every input track.
    """
    links: list[tuple[float, int, int]] = []
    for a in summaries:
        for b in summaries:
            if a is b or a.cls != b.cls:
                continue
            gap = (b.first_frame - a.last_frame) / fps
            if gap <= 0 or gap > max_gap_s:
                continue
            if a.team >= 0 and b.team >= 0 and a.team != b.team:
                continue
            dist = float(np.linalg.norm(b.first_xy - a.last_xy))
            if dist > max_speed_mps * gap + slack_m:
                continue
            links.append((dist / max(gap, 1e-3), a.track_id, b.track_id))
    links.sort()

    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for _cost, a, b in links:
        # One successor / predecessor per fragment; since every link goes
        # forward in time, the resulting chains never overlap themselves.
        if a in succ or b in pred:
            continue
        succ[a], pred[b] = b, a

    mapping: dict[int, int] = {}
    for s in summaries:
        root = s.track_id
        while root in pred:
            root = pred[root]
        mapping[s.track_id] = root
    return mapping
