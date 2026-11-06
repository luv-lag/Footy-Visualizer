"""camera.py -- track the broadcast camera's pitch homography through a shot.

Calibrating every frame from scratch is too slow (pitchcal: seconds per
frame) and jittery, and single-frame calibrators honestly refuse frames
with too little visible paint.  Within one continuous shot the camera
moves smoothly, so the homography can instead be *tracked*:

1. **Anchor** -- a full single-frame calibration (pitchcal, optionally the
   PnLCalib fallback) on the first frame of the shot, retried every
   ``anchor_retry_every`` frames until one succeeds.
2. **Propagate** -- every frame, estimate the inter-frame image homography
   G from sparse optical flow on grass pixels (players masked out, since
   they move independently of the pitch) and update ``H_t = G @ H_{t-1}``.
3. **Refine** -- every ``refine_every`` frames, polish the propagated H
   against that frame's own detected markings (pitchcal's distance-field
   refinement).  This is what stops chained flow from drifting; the
   resulting fit doubles as a health check.
4. **Re-anchor / give up** -- if refinement says the propagated H no
   longer explains the markings, try a fresh anchor; if that also fails,
   the tracker reports itself *lost* (no homography) rather than drawing
   a plausible-looking but wrong minimap.

Coordinates follow pitchcal: ``H`` maps corner-origin pitch metres to
*undistorted* image pixels; ``dm`` undistorts raw pixels first.

Orientation continuity: a fresh anchor is canonical per frame (camera on
the y < 0 side), which for a single camera is consistent across the shot,
but the anchor is still compared with the propagated H and its rot180
twin, and the closer one is kept -- so the minimap can never swap ends
mid-shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from pitchcal import DivisionModel, PitchModel, calibrate
from pitchcal.detect import detect_features, field_mask
from pitchcal.geometry import apply_h_w
from pitchcal.solve import Candidate, Scorer, mapping_discrepancy_m, refine_candidate

#: An anchor calibrator: (frame, player_boxes, player_feet, player_labels)
#: -> (H, dm, source_name) or None when it refuses the frame.
AnchorFn = Callable[
    [np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None],
    "tuple[np.ndarray, DivisionModel, str] | None",
]


@dataclass
class CameraState:
    """The camera model for one frame.

    Attributes:
        frame_idx: index of the frame in the source video.
        H: pitch (corner-origin metres) -> undistorted image homography, or
            None when the tracker is lost / not yet anchored.
        dm: lens-distortion model to apply to raw pixels before ``H^-1``.
        source: how H was obtained -- "anchor:<calibrator>", "refined",
            "flow", or "lost".
        fit: pitchcal's normalised marking fit at the last check (lower is
            better; NaN when this frame was not scored).
        explained: fraction of detected marking pixels the template explains
            at the last check (NaN when not scored).
    """

    frame_idx: int
    H: np.ndarray | None
    dm: DivisionModel | None
    source: str
    fit: float = float("nan")
    explained: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.H is not None

    def image_to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        """Raw (distorted) pixel coords (N, 2) -> pitch metres (N, 2)."""
        if self.H is None:
            raise ValueError("no homography for this frame")
        pts = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        und = self.dm.undistort(pts) if self.dm is not None else pts
        p, _ = apply_h_w(np.linalg.inv(self.H), und)
        return p

    def pitch_to_image(self, pts_m: np.ndarray) -> np.ndarray:
        """Pitch metres (N, 2) -> raw (distorted) pixel coords (N, 2)."""
        if self.H is None:
            raise ValueError("no homography for this frame")
        p, _ = apply_h_w(self.H, np.asarray(pts_m, dtype=float).reshape(-1, 2))
        return self.dm.distort(p) if self.dm is not None else p


def pitchcal_anchor(
    frame: np.ndarray,
    player_boxes: np.ndarray | None,
    player_feet: np.ndarray | None,
    player_labels: np.ndarray | None,
) -> tuple[np.ndarray, DivisionModel, str] | None:
    """Default anchor: pitchcal's single-frame calibration (CPU only)."""
    res = calibrate(
        frame, player_feet=player_feet, player_labels=player_labels, player_boxes=player_boxes
    )
    if not res.ok:
        return None
    return res.H, res.dm, "pitchcal"


def pitchcal_then_pnlcalib_anchor(device: str = "cuda:0") -> AnchorFn:
    """Anchor with pitchcal, falling back to PnLCalib (needs the torch venv)."""

    def anchor(frame, player_boxes, player_feet, player_labels):
        out = pitchcal_anchor(frame, player_boxes, player_feet, player_labels)
        if out is not None:
            return out
        from frame_analysis import pnlcalib_corner_homography  # noqa: PLC0415

        H, _detail = pnlcalib_corner_homography(frame, device)
        if H is None:
            return None
        h, w = frame.shape[:2]
        return H, DivisionModel(0.0, (w, h)), "pnlcalib"

    return anchor


class ManualAnchor:
    """Anchor from hand-picked landmarks on one frame (for fields pitchcal can't read).

    Multi-sport turf is the usual case: American-football yard lines and
    numbers dominate the paint, and the soccer markings are faint or
    coloured.  The JSON file holds ``frame`` (the frame the points were
    picked on), ``image_points`` (N >= 4 raw pixel coords) and
    ``pitch_points`` (the same N points in corner-origin pitch metres).
    The tracker sets :attr:`current` to the frame index before each update;
    every other frame is refused, so flow carries the camera from there.
    """

    def __init__(self, path: str) -> None:
        import json  # noqa: PLC0415

        with open(path) as f:
            spec = json.load(f)
        self.frame = int(spec["frame"])
        img = np.asarray(spec["image_points"], dtype=float)
        pit = np.asarray(spec["pitch_points"], dtype=float)
        if len(img) < 4 or img.shape != pit.shape:
            raise ValueError(f"{path}: need >= 4 matching image_points / pitch_points")
        self.H, _ = cv2.findHomography(pit, img, 0)
        self.current: int | None = None

    def __call__(self, frame, player_boxes, player_feet, player_labels):
        if self.current != self.frame:
            return None
        h, w = frame.shape[:2]
        return self.H, DivisionModel(0.0, (w, h)), "manual"


class CameraTracker:
    """Online camera tracker for ONE continuous shot (call :meth:`reset` at cuts)."""

    def __init__(
        self,
        anchor_fn: AnchorFn = pitchcal_anchor,
        pitch: PitchModel | None = None,
        refine_every: int = 5,
        reanchor_every: int = 0,
        anchor_retry_every: int = 10,
        max_fit: float = 0.40,
        min_explained: float = 0.35,
        min_flow_inliers: int = 25,
        max_backfill: int = 250,
    ) -> None:
        """
        Args:
            anchor_fn: full single-frame calibrator used to (re)anchor.
            pitch: pitch template (defaults to FIFA 105 x 68).
            refine_every: frames between marking-based refinements (1 =
                every frame; each costs ~0.3 s on CPU at 720p).
            reanchor_every: frames between periodic re-anchors (0 = only
                when lost).  A re-anchor replaces the tracked H only if it
                explains the frame's pitch markings better.
            anchor_retry_every: while lost, frames between anchor attempts.
            max_fit / min_explained: health gates on a refined H, same
                semantics as pitchcal's acceptance thresholds.
            min_flow_inliers: RANSAC inliers needed to trust a flow step.
            max_backfill: how many un-anchored frames to remember so that,
                once an anchor succeeds, flow can be run *backwards* to
                recover them (see :meth:`pop_backfilled`).  0 disables it.
        """
        self.anchor_fn = anchor_fn
        self.pitch = pitch or PitchModel()
        self.refine_every = max(1, refine_every)
        self.reanchor_every = reanchor_every
        self.anchor_retry_every = max(1, anchor_retry_every)
        self.max_fit = max_fit
        self.min_explained = min_explained
        self.min_flow_inliers = min_flow_inliers
        self.max_backfill = max_backfill
        self.reset()

    def reset(self) -> None:
        """Forget all state (start of a new shot)."""
        self.H: np.ndarray | None = None
        self.dm: DivisionModel | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._since_refine = 0
        self._since_anchor = 0
        self._since_attempt = self.anchor_retry_every  # attempt immediately
        self._pending: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._backfilled: list[CameraState] = []

    def pop_backfilled(self) -> list[CameraState]:
        """States recovered for earlier lost frames since the last call.

        When the tracker anchors after a run of lost frames (typically the
        start of a shot, before the first frame pitchcal accepts), it
        propagates the new anchor *backwards* through those frames with the
        same optical flow.  The recovered states (source "backfill") should
        replace the "lost" states previously returned for those frames.
        """
        out, self._backfilled = self._backfilled, []
        return out

    # ------------------------------------------------------------------ #
    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        player_boxes: np.ndarray | None = None,
        player_feet: np.ndarray | None = None,
        player_labels: np.ndarray | None = None,
    ) -> CameraState:
        """Advance the tracker by one frame and return that frame's camera.

        Args:
            frame: BGR frame.
            frame_idx: its index in the source video (bookkeeping only).
            player_boxes: optional (N, 4) xyxy person boxes; masked out of
                the flow and marking evidence.
            player_feet / player_labels: optional anchor hints (pitchcal's
                formation fallback).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._flow_mask(frame, player_boxes)
        diag: dict = {}
        source = "lost"
        fit = explained = float("nan")

        if self.H is not None and self._prev_gray is not None:
            G, n_in = self._flow_step(self._prev_gray, self._prev_mask, gray)
            diag["flow_inliers"] = n_in
            if G is None:
                self._lose("optical flow failed", diag)
            else:
                self.H = _normalize(G @ self.H)
                source = "flow"
                self._since_refine += 1
                self._since_anchor += 1

        if self.H is not None and self._since_refine >= self.refine_every:
            ok, fit, explained = self._refine(frame, player_boxes)
            self._since_refine = 0
            if ok:
                source = "refined"
            else:
                self._lose(f"drift: fit={fit:.2f} explained={explained:.2f}", diag)

        want_anchor = self.H is None and self._since_attempt >= self.anchor_retry_every
        if self.reanchor_every and self._since_anchor >= self.reanchor_every:
            want_anchor = True
        if want_anchor:
            if self._try_anchor(frame, player_boxes, player_feet, player_labels, diag):
                source = diag["anchor_source"]
                self._backfill(gray, mask)
        elif self.H is None:
            self._since_attempt += 1

        self._prev_gray, self._prev_mask = gray, mask
        if self.H is None:
            source = "lost"
            if self.max_backfill:
                self._pending.append((frame_idx, gray, mask))
                del self._pending[: -self.max_backfill]
        return CameraState(
            frame_idx=frame_idx,
            H=None if self.H is None else self.H.copy(),
            dm=self.dm,
            source=source,
            fit=fit,
            explained=explained,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------ #
    def _lose(self, reason: str, diag: dict) -> None:
        self.H = None
        diag["lost_reason"] = reason
        self._since_attempt = self.anchor_retry_every  # re-anchor right away

    def _backfill(self, gray: np.ndarray, mask: np.ndarray) -> None:
        """Propagate the fresh anchor backwards over the pending lost frames."""
        H, prev_gray, prev_mask = self.H, gray, mask
        for idx, g, m in reversed(self._pending):
            G, _ = self._flow_step(prev_gray, prev_mask, g)
            if G is None:
                break  # e.g. a cut: frames before it belong to another shot
            H = _normalize(G @ H)
            self._backfilled.append(CameraState(idx, H.copy(), self.dm, "backfill"))
            prev_gray, prev_mask = g, m
        self._backfilled.reverse()
        self._pending = []

    def _try_anchor(self, frame, boxes, feet, labels, diag) -> bool:
        self._since_attempt = 0
        out = self.anchor_fn(frame, boxes, feet, labels)
        if out is None:
            diag["anchor"] = "refused"
            return False
        H_new, dm_new, name = out
        H_new = _normalize(H_new)
        if self.H is not None:
            # Orientation continuity: keep whichever of the anchor and its
            # rot180 twin agrees with the tracked H.
            size = (frame.shape[1], frame.shape[0])
            twin = _normalize(H_new @ self.pitch.rot180())
            d = mapping_discrepancy_m(self.H, H_new, size)
            d_twin = mapping_discrepancy_m(self.H, twin, size)
            if d_twin < d:
                H_new, d = twin, d_twin
                diag["anchor_flipped_for_continuity"] = True
            diag["anchor_jump_m"] = round(float(d), 3)
            # Let this frame's markings decide: neither a fresh single-frame
            # anchor (occasionally ~1 m off) nor a long-tracked H (slow drift
            # that refinement can lock onto) is trusted by default.
            s_track, s_new = self._marking_scores(frame, boxes, [(self.H, self.dm), (H_new, dm_new)])
            diag["anchor_scores"] = {"track": round(s_track, 4), "anchor": round(s_new, 4)}
            if not s_new < s_track:
                self._since_anchor = 0
                diag["anchor"] = "kept track: fits this frame's markings better"
                return False
        self.H, self.dm = H_new, dm_new
        self._since_anchor = 0
        self._since_refine = 0
        diag["anchor"] = "ok"
        diag["anchor_source"] = f"anchor:{name}"
        return True

    def _marking_scores(self, frame, boxes, cams) -> list[float]:
        """pitchcal's candidate score (fit + unexplained paint) per (H, dm)."""
        features = detect_features(frame, player_boxes=boxes)
        scorer = Scorer(features, self.pitch)
        out = []
        for H, dm in cams:
            if dm is not None and abs(features.dm.lam - dm.lam) > 1e-6:
                H = _rebase_distortion(H, dm, features.dm, features.image_size)
            fit, _ = scorer.fit(H)
            out.append(float(fit + 0.6 * (1.0 - scorer.explained(H))) if np.isfinite(fit) else float("inf"))
        return out

    def _refine(self, frame, boxes) -> tuple[bool, float, float]:
        features = detect_features(frame, player_boxes=boxes)
        if len(features.line_points) < 50:
            return False, float("inf"), 0.0
        # Detected features carry this frame's own distortion estimate; the
        # tracked H lives in the previous model's undistorted space, so
        # re-express it before scoring.
        H = self.H
        if self.dm is not None and abs(features.dm.lam - self.dm.lam) > 1e-6:
            H = _rebase_distortion(H, self.dm, features.dm, features.image_size)
        scorer = Scorer(features, self.pitch)
        fit0, n_in = scorer.fit(H)
        cand = Candidate(H=H, strategy="track", assignment={}, dlt_residual=0.0,
                         n_constraints=0, fit=fit0, n_inside=n_in)
        cand = refine_candidate(cand, scorer, features)
        explained = scorer.explained(cand.H)
        if not np.isfinite(cand.fit) or cand.fit > self.max_fit or explained < self.min_explained:
            return False, float(cand.fit), float(explained)
        self.H, self.dm = _normalize(cand.H), features.dm
        return True, float(cand.fit), float(explained)

    def _flow_mask(self, frame, boxes) -> np.ndarray:
        mask = field_mask(frame)
        if boxes is not None:
            for x0, y0, x1, y1 in np.asarray(boxes, dtype=float).reshape(-1, 4):
                pad = 0.15 * (x1 - x0)
                cv2.rectangle(mask, (int(x0 - pad), int(y0 - pad)),
                              (int(x1 + pad), int(y1 + pad)), 0, -1)
        return mask

    def _flow_step(self, prev, prev_mask, curr) -> tuple[np.ndarray | None, int]:
        """Inter-frame homography (undistorted pixels) from sparse LK flow."""
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.005,
                                      minDistance=8, mask=prev_mask, blockSize=7)
        if pts is None or len(pts) < self.min_flow_inliers:
            return None, 0
        lk = dict(winSize=(21, 21), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, **lk)
        fb_err = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)
        if good.sum() < self.min_flow_inliers:
            return None, int(good.sum())
        p0 = pts.reshape(-1, 2)[good]
        p1 = nxt.reshape(-1, 2)[good]
        if self.dm is not None:
            p0, p1 = self.dm.undistort(p0), self.dm.undistort(p1)
        G, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 1.5, maxIters=2000, confidence=0.995)
        n_in = 0 if inl is None else int(inl.sum())
        if G is None or n_in < self.min_flow_inliers:
            return None, n_in
        return G, n_in


def _normalize(H: np.ndarray) -> np.ndarray:
    return H / np.linalg.norm(H)


def _rebase_distortion(H, dm_old, dm_new, image_size) -> np.ndarray:
    """Re-express H (in dm_old's undistorted space) in dm_new's space."""
    w, h = image_size
    xs, ys = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 7), np.linspace(0.05 * h, 0.95 * h, 5))
    raw = np.stack([xs.ravel(), ys.ravel()], axis=1)
    A = cv2.findHomography(dm_old.undistort(raw), dm_new.undistort(raw))[0]
    return _normalize(A @ H) if A is not None else H


# ---------------------------------------------------------------------- #
# Offline smoothing
# ---------------------------------------------------------------------- #
def smooth_states(states: list[CameraState], window: int = 9) -> list[CameraState]:
    """Temporally smooth tracked homographies (offline, centred window).

    Parametrisation-free: within each run of consecutive valid states,
    a fixed grid of image points is mapped to the pitch, those pitch points
    are averaged over a centred window, and H is re-fitted to them.  Runs
    are never smoothed across a lost frame (cut / tracking failure).
    """
    if window < 3 or not states:
        return states
    out = list(states)
    half = window // 2
    i = 0
    while i < len(states):
        if not states[i].ok:
            i += 1
            continue
        j = i
        while j + 1 < len(states) and states[j + 1].ok:
            j += 1
        run = states[i : j + 1]
        if len(run) >= 3:
            w, h = _grid_size(run[0])
            xs, ys = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 5), np.linspace(0.3 * h, 0.95 * h, 4))
            img_pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
            und = [s.dm.undistort(img_pts) if s.dm is not None else img_pts for s in run]
            pitch_pts = np.stack([apply_h_w(np.linalg.inv(s.H), u)[0] for s, u in zip(run, und)])
            for k, s in enumerate(run):
                # Symmetric window, shrunk at run edges: a one-sided window
                # would bias the ends towards the neighbouring frames.
                r = min(half, k, len(run) - 1 - k)
                lo, hi = k - r, k + r + 1
                avg = pitch_pts[lo:hi].mean(axis=0)
                H_s, _ = cv2.findHomography(avg, und[k])
                if H_s is not None:
                    out[i + k] = CameraState(
                        s.frame_idx, _normalize(H_s), s.dm, s.source, s.fit, s.explained,
                        s.diagnostics,
                    )
        i = j + 1
    return out


def _grid_size(state: CameraState) -> tuple[int, int]:
    if state.dm is not None:
        return state.dm.w, state.dm.h
    raise ValueError("state has no image size (dm missing)")
