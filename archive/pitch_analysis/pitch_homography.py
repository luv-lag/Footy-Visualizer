#!/usr/bin/env python3
"""
Pitch Homography + Relative-Confidence Layer
============================================
Isolated broadcast still → pitch-keypoint homography → players in rectified
pitch space → a *pairwise relative-confidence* layer for downstream pass /
dribble / shot deliverables.

Design contract
---------------
  * No metric accuracy is claimed. The target rectangle is a *consistent model*
    (a normalised pitch template in template-pixels = cm / 10), NOT meters.
    Relative + self-consistent is the goal.
  * HARD REQUIREMENT: never emit a confident relation between two players when
    the geometry does not support it. This is enforced two ways:
      1. ABSTAIN frame grade → all pairwise confidences are hard-zeroed.
      2. Pairwise confidence is governed by the *less-trusted* endpoint
         (min of the two hull-trusts), so one badly-extrapolated player drags
         every relation it participates in toward zero.

  * solve_homography() and grade_frame() are pure functions of whatever
    correspondences arrive — no image, no temporal state. A future video
    version pools keypoints across frames *before* solve_homography() and the
    solver + grader are reused untouched.

Keypoint source
---------------
  Roboflow hosted inference, model `football-field-detection-f07vi/14`
  (32-vertex SoccerPitchConfiguration order). The hosted REST response is
  named-points-plus-confidences and a *different shape* than the detection-box
  response — it is parsed by _parse_keypoint_response() ALONE. A later swap to
  local .pt weights touches only that one parser; everything downstream consumes
  list[NamedKeypoint].

  Verified empirically (verify_pitch_keypoints.py): the endpoint returns real,
  varying per-point confidences (not coordinates-only, not all-1.0). The grader
  keys entirely off those.

Usage
-----
    ROBOFLOW_API_KEY=<key> python pitch_homography.py brasilscotland2.png
"""

from __future__ import annotations
import base64
import os
import sys
import argparse
import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
import requests

from football_analysis import detect_all, DEFAULT_MODEL_ID, TEAM_BGR
from vanishing_point import assign_teams_2d

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# PITCH TEMPLATE  (vendored from roboflow/sports SoccerPitchConfiguration)
# ══════════════════════════════════════════════════════════════════════════════
# Vendored rather than importing `sports` so this module keeps the same minimal
# dependency footprint as football_analysis.py (cv2 / numpy / requests only).
# Vertices are in natural order: index i == pitch vertex (i+1), matching the
# `class` field "01".."32" returned per keypoint, so we map by class string and
# never depend on positional ordering of the (variable-length) REST response.

@dataclass
class SoccerPitchConfiguration:
    width: int = 7000          # [cm]
    length: int = 12000        # [cm]
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> list[tuple[float, float]]:
        w, l = self.width, self.length
        pb, pl = self.penalty_box_width, self.penalty_box_length
        gb, gl = self.goal_box_width, self.goal_box_length
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        return [
            (0, 0), (0, (w - pb) / 2), (0, (w - gb) / 2), (0, (w + gb) / 2),
            (0, (w + pb) / 2), (0, w),
            (gl, (w - gb) / 2), (gl, (w + gb) / 2),
            (ps, w / 2),
            (pl, (w - pb) / 2), (pl, (w - gb) / 2), (pl, (w + gb) / 2),
            (pl, (w + pb) / 2),
            (l / 2, 0), (l / 2, w / 2 - r), (l / 2, w / 2 + r), (l / 2, w),
            (l - pl, (w - pb) / 2), (l - pl, (w - gb) / 2),
            (l - pl, (w + gb) / 2), (l - pl, (w + pb) / 2),
            (l - ps, w / 2),
            (l - gl, (w - gb) / 2), (l - gl, (w + gb) / 2),
            (l, 0), (l, (w - pb) / 2), (l, (w - gb) / 2), (l, (w + gb) / 2),
            (l, (w + pb) / 2), (l, w),
            (l / 2 - r, w / 2), (l / 2 + r, w / 2),
        ]

    edges: list[tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])


CONFIG = SoccerPitchConfiguration()
PITCH_MODEL_ID = "football-field-detection-f07vi/14"
INFER_URL = "https://detect.roboflow.com"
INFER_TIMEOUT = 30

# Template pixels = cm / TEMPLATE_SCALE. The target "consistent rectangle".
TEMPLATE_SCALE = 10.0
TEMPLATE_LEN_PX = CONFIG.length / TEMPLATE_SCALE   # 1200
TEMPLATE_WID_PX = CONFIG.width / TEMPLATE_SCALE    # 700

# Centre-circle vertices (1-based class numbers): the two halfway-line/circle
# intersections (15, 16) and the two circle tangent points on the halfway line
# (31, 32). They are ordinary keypoints used automatically when confident; see
# the ellipse-fit note in solve_homography().
CENTRE_CIRCLE_CLASSES = {15, 16, 31, 32}


def _template_xy(vertex_idx0: int) -> np.ndarray:
    """Pitch vertex (0-based) → template-pixel coordinate."""
    cx, cy = CONFIG.vertices[vertex_idx0]
    return np.array([cx / TEMPLATE_SCALE, cy / TEMPLATE_SCALE], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NamedKeypoint:
    index: int             # 0-based vertex index into CONFIG.vertices
    label: str             # class string as returned, e.g. "01".."32"
    image_xy: np.ndarray   # (2,) float32 — pixel coords in the input image
    pitch_xy: np.ndarray   # (2,) float32 — template-pixel coords (consistent model)
    conf: float            # model confidence [0, 1]


@dataclass
class HomographyResult:
    H: np.ndarray | None       # (3,3) image→template, or None when solve failed
    inlier_mask: np.ndarray    # (N,) bool from RANSAC; empty when H is None
    n_inliers: int
    mean_reproj_px: float      # mean reprojection error on inliers, in IMAGE px
    src_pts: np.ndarray        # (N,2) image keypoints fed to the solver
    dst_pts: np.ndarray        # (N,2) template keypoints fed to the solver
    used_classes: list[int]    # 1-based vertex numbers fed to the solver
    centre_circle_inliers: int # how many centre-circle pts survived as inliers


class Grade(str, Enum):
    SOLID = "solid"
    PARTIAL = "partial"
    ABSTAIN = "abstain"        # a correct, first-class output


@dataclass
class FrameGrade:
    grade: Grade
    n_above_conf: int          # keypoints passing conf threshold (pre-RANSAC)
    n_inliers: int
    mean_reproj_px: float
    hull_pts: np.ndarray       # (K,2) convex hull of inlier image keypoints
    hull_area_frac: float      # hull area / image area — coverage proxy
    hull_scale_px: float       # sqrt(hull area) — characteristic hull size


@dataclass
class PlayerGroundPoint:
    player_idx: int
    team: int
    image_xy: np.ndarray            # (2,) bottom-centre of bbox (the ground point)
    pitch_xy: np.ndarray | None     # (2,) template px; None when grade == ABSTAIN
    hull_trust: float               # [0,1]; 1.0 strictly inside hull, smooth decay out
    signed_dist_px: float           # raw signed dist to hull (px); <0 in, >0 out


@dataclass
class PairwiseConf:
    i: int
    j: int
    conf: float                     # [0,1]; hard 0 when grade == ABSTAIN
    trust_i: float
    trust_j: float


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — KEYPOINT DETECTION  (Roboflow hosted; parser isolated)
# ══════════════════════════════════════════════════════════════════════════════

def _encode_image(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def _parse_keypoint_response(data: dict, conf_threshold: float) -> list[NamedKeypoint]:
    """
    ISOLATED parser for the hosted keypoint REST shape. The ONLY function that
    understands Roboflow's keypoint JSON; a local-.pt swap reimplements just this.

    Shape: data["predictions"] is a list of pitch detections, each with a nested
    "keypoints" list of {x, y, confidence, class, class_id}. The REST response is
    variable-length (e.g. 27 of 32 points).

    IDENTITY IS `class_id`, NOT the `class` string. class_id is the model's
    keypoint slot, which aligns positionally with CONFIG.vertices (vertices are
    stored in slot order). The `class` string is a cosmetic display name drawn
    from a *permuted* labels list (e.g. slot 30 is labelled "14" but is
    geometrically the centre-circle left tangent) — using it as the vertex index
    scrambles the correspondences and the homography blows up (verified: ~307 px
    vs ~10 px mean reprojection). Keep `class` only as a human-readable label.
    """
    preds = data.get("predictions", [])
    if not preds:
        return []
    # Single pitch per frame; if several, take the highest-confidence detection.
    pred = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
    out: list[NamedKeypoint] = []
    for k in pred.get("keypoints", []):
        try:
            idx0 = int(k["class_id"])           # model slot == vertex index
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= idx0 <= 31):
            continue
        conf = float(k.get("confidence", 0.0))
        if conf < conf_threshold:
            continue
        out.append(NamedKeypoint(
            index=idx0,
            label=str(k.get("class", f"{idx0:02d}")),
            image_xy=np.array([float(k["x"]), float(k["y"])], dtype=np.float32),
            pitch_xy=_template_xy(idx0),
            conf=conf,
        ))
    return out


def detect_pitch_keypoints(
    image_bgr: np.ndarray,
    api_key: str | None = None,
    model_id: str = PITCH_MODEL_ID,
    conf_threshold: float = 0.5,
) -> list[NamedKeypoint]:
    """
    Returns keypoints with model confidence >= conf_threshold.
    Clean stub: prints one warning and returns [] when no key / call fails.
    (An empty list flows downstream to H=None → ABSTAIN, the correct behaviour.)
    """
    key = (api_key or os.environ.get("ROBOFLOW_API_KEY", "")).strip()
    if not key:
        print("[WARN] No ROBOFLOW_API_KEY — pitch keypoints unavailable (stub → []).")
        return []
    parts = model_id.split("/")
    endpoint = f"{INFER_URL}/{parts[0]}/{parts[1]}" if len(parts) == 2 \
        else f"{INFER_URL}/{parts[1]}/{parts[2]}"
    try:
        resp = requests.post(
            endpoint, params={"api_key": key}, data=_encode_image(image_bgr),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=INFER_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_keypoint_response(resp.json(), conf_threshold)
    except Exception as exc:
        print(f"[WARN] Pitch keypoint inference failed ({exc}); returning [].")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — HOMOGRAPHY  (pure function of correspondences)
# ══════════════════════════════════════════════════════════════════════════════

def solve_homography(
    keypoints: list[NamedKeypoint],
    ransac_reproj_threshold: float = 8.0,
) -> HomographyResult:
    """
    Estimate the image→template homography from confident keypoints via
    cv2.findHomography + RANSAC against the consistent rectangle model.

    Pure function of the correspondences: a video version pools NamedKeypoints
    across frames before calling this and reuses it unchanged.

    ransac_reproj_threshold is in TEMPLATE pixels (the destination space).
    Reprojection error is reported back in IMAGE pixels (via the inverse map),
    which is the meaningful unit for the grader's thresholds.

    Centre-circle points (classes 15/16/31/32) are used like any other vertex
    when present and confident. NOTE: on sparse frames an explicit ellipse fit
    to the centre-circle arc (cv2.fitEllipse on detected circle pixels, then
    sampling tangent/intersection points) would add 2-4 extra correspondences
    and stabilise the solve — a clean future extension that would slot in here
    by appending synthesised NamedKeypoints before this call.
    """
    empty = np.array([], dtype=bool)
    if len(keypoints) < 4:
        return HomographyResult(None, empty, 0, float("inf"),
                                np.empty((0, 2), np.float32),
                                np.empty((0, 2), np.float32), [], 0)

    src = np.array([kp.image_xy for kp in keypoints], dtype=np.float32)
    dst = np.array([kp.pitch_xy for kp in keypoints], dtype=np.float32)
    used_classes = [kp.index + 1 for kp in keypoints]

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
    if H is None:
        return HomographyResult(None, empty, 0, float("inf"),
                                src, dst, used_classes, 0)

    mask_bool = mask.ravel().astype(bool)
    n_inliers = int(mask_bool.sum())

    # Reprojection error in IMAGE pixels: map template inliers back via H_inv.
    try:
        H_inv = np.linalg.inv(H)
        dst_in = dst[mask_bool].reshape(-1, 1, 2)
        img_pred = cv2.perspectiveTransform(dst_in, H_inv).reshape(-1, 2)
        mean_reproj = float(np.linalg.norm(img_pred - src[mask_bool], axis=1).mean())
    except np.linalg.LinAlgError:
        mean_reproj = float("inf")

    cc_inliers = sum(
        1 for kp, keep in zip(keypoints, mask_bool)
        if keep and (kp.index + 1) in CENTRE_CIRCLE_CLASSES
    )

    return HomographyResult(H, mask_bool, n_inliers, mean_reproj,
                            src, dst, used_classes, cc_inliers)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FRAME GRADE  (the centerpiece)
# ══════════════════════════════════════════════════════════════════════════════

def grade_frame(
    homography: HomographyResult,
    image_shape: tuple[int, int],
    solid_min_inliers: int = 6,
    solid_max_reproj_px: float = 8.0,
    partial_min_inliers: int = 4,
    partial_max_reproj_px: float = 15.0,
) -> FrameGrade:
    """
    Decide SOLID / PARTIAL / ABSTAIN from inlier count + reprojection error,
    and build the convex hull of the inlier image keypoints (the region inside
    which player positions are interpolated, not extrapolated).

    ABSTAIN is a correct output: too few inliers or too much reprojection error
    means the geometry cannot support confident relations, so we say so.
    """
    H_img, W_img = image_shape[:2]
    img_area = float(H_img * W_img)

    def _abstain() -> FrameGrade:
        return FrameGrade(Grade.ABSTAIN, 0, homography.n_inliers,
                          homography.mean_reproj_px,
                          np.empty((0, 2), np.float32), 0.0, 0.0)

    if homography.H is None or homography.n_inliers < partial_min_inliers \
            or homography.mean_reproj_px > partial_max_reproj_px:
        fg = _abstain()
        fg.n_above_conf = len(homography.src_pts)
        return fg

    inlier_img = homography.src_pts[homography.inlier_mask].astype(np.float32)
    hull = cv2.convexHull(inlier_img).reshape(-1, 2)
    hull_area = float(cv2.contourArea(hull.astype(np.float32)))
    hull_area_frac = hull_area / img_area if img_area > 0 else 0.0
    hull_scale = float(np.sqrt(max(hull_area, 1.0)))

    if homography.n_inliers >= solid_min_inliers \
            and homography.mean_reproj_px <= solid_max_reproj_px:
        grade = Grade.SOLID
    else:
        grade = Grade.PARTIAL

    return FrameGrade(grade, len(homography.src_pts), homography.n_inliers,
                      homography.mean_reproj_px, hull, hull_area_frac, hull_scale)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — HULL TRUST  (smooth extrapolation falloff)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hull_trust(
    query_pts: np.ndarray,
    hull_pts: np.ndarray,
    knee_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-point trust as a smooth falloff PAST the hull boundary.

    Interior points are interpolated — there is no extrapolation error inside —
    so trust is clipped to exactly 1.0 for any point strictly inside (signed
    distance < 0). The smooth decay models extrapolation error and applies only
    at and beyond the boundary:

        d <= 0 (inside) : trust = 1.0
        d  > 0 (outside): trust = 2 ** (-(d / knee_px) ** 2)

    This is continuous and flat-topped at the edge (trust ≈ 1.0 at d = 0, the
    "knee" half-trust point sits OUT at d = knee_px, never centred on the
    boundary). `knee_px` is tied to real scale by the caller (reprojection error
    and/or a fraction of hull scale) — never a hardcoded constant.

    Returns (trust, signed_dist_px) where signed_dist_px is the RAW signed
    distance (negative inside, positive outside) retained for debugging partial
    frames; trust is the clipped value for use at the call site.
    """
    n = len(query_pts)
    if n == 0:
        return np.array([]), np.array([])
    if len(hull_pts) < 3:
        # No usable hull → everything is "outside"; trust 0, distance unknown.
        return np.zeros(n), np.full(n, np.inf)

    contour = hull_pts.reshape(-1, 1, 2).astype(np.float32)
    knee = max(float(knee_px), 1.0)
    trust = np.empty(n)
    signed = np.empty(n)
    for i, pt in enumerate(query_pts):
        # pointPolygonTest: + inside, - outside. Flip so + means outside.
        d_out = -cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), True)
        signed[i] = d_out
        trust[i] = 1.0 if d_out <= 0 else 2.0 ** (-(d_out / knee) ** 2)
    return trust, signed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PLAYER MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def _ground_point(bbox: list[int]) -> np.ndarray:
    """Bottom-centre of the bbox — where the player meets the pitch."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float32)


def map_players(
    players: list[dict],
    team_labels: np.ndarray,
    homography: HomographyResult,
    frame_grade: FrameGrade,
    hull_knee_frac: float = 0.05,
) -> list[PlayerGroundPoint]:
    """
    Map each player's ground point into template space and attach hull-trust.

    pitch_xy is None for every player when grade == ABSTAIN (no trustworthy
    geometry). hull_trust + raw signed distance are computed regardless, so a
    partial/abstain frame can still be debugged.

    The trust knee width is tied to real scale: max(mean reproj error,
    hull_knee_frac * hull scale).
    """
    ground = np.array([_ground_point(p["bbox"]) for p in players], dtype=np.float32)

    if frame_grade.grade == Grade.ABSTAIN or homography.H is None:
        return [
            PlayerGroundPoint(i, int(team_labels[i]), ground[i], None, 0.0, float("inf"))
            for i in range(len(players))
        ]

    knee_px = max(frame_grade.mean_reproj_px,
                  hull_knee_frac * frame_grade.hull_scale_px)
    trust, signed = compute_hull_trust(ground, frame_grade.hull_pts, knee_px)

    pitch = cv2.perspectiveTransform(
        ground.reshape(-1, 1, 2), homography.H).reshape(-1, 2)

    return [
        PlayerGroundPoint(i, int(team_labels[i]), ground[i],
                          pitch[i].astype(np.float32),
                          float(trust[i]), float(signed[i]))
        for i in range(len(players))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PAIRWISE RELATIVE CONFIDENCE  (the deliverable)
# ══════════════════════════════════════════════════════════════════════════════

_GRADE_FACTOR = {Grade.SOLID: 1.0, Grade.PARTIAL: 0.6, Grade.ABSTAIN: 0.0}


def pairwise_confidence(
    ground_points: list[PlayerGroundPoint],
    frame_grade: FrameGrade,
) -> list[PairwiseConf]:
    """
    Relative confidence in the geometric relation between each player pair.

        conf(i, j) = grade_factor(frame) * min(trust_i, trust_j)

    The min() makes the LESS-TRUSTED endpoint govern the relation — a single
    badly-extrapolated player drags every pair it touches toward zero. Combined
    with grade_factor == 0 on ABSTAIN, this satisfies the hard requirement:
    no confident relation is ever emitted where the geometry doesn't support it.
    """
    factor = _GRADE_FACTOR[frame_grade.grade]
    out: list[PairwiseConf] = []
    n = len(ground_points)
    for a in range(n):
        for b in range(a + 1, n):
            ta = ground_points[a].hull_trust
            tb = ground_points[b].hull_trust
            out.append(PairwiseConf(
                ground_points[a].player_idx, ground_points[b].player_idx,
                factor * min(ta, tb), ta, tb,
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def viz_keypoints_on_frame(
    image_bgr: np.ndarray,
    keypoints: list[NamedKeypoint],
    homography: HomographyResult,
) -> np.ndarray:
    """Overlay keypoints on the still: green = RANSAC inlier, red = outlier;
    radius scales with confidence; class label printed."""
    out = image_bgr.copy()
    inlier_set = set()
    if homography.H is not None and len(homography.inlier_mask):
        inlier_set = {i for i, keep in enumerate(homography.inlier_mask) if keep}
    for i, kp in enumerate(keypoints):
        c = (int(kp.image_xy[0]), int(kp.image_xy[1]))
        color = (60, 220, 60) if i in inlier_set else (60, 60, 230)
        rad = int(4 + 8 * kp.conf)
        cv2.circle(out, c, rad, color, -1, cv2.LINE_AA)
        cv2.circle(out, c, rad, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{kp.label}:{kp.conf:.2f}", (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _draw_pitch_template(scale: float = 1.0, margin: int = 20) -> tuple[np.ndarray, float, int]:
    """Top-down pitch canvas with line markings, scaled from the template."""
    W = int(TEMPLATE_LEN_PX * scale) + 2 * margin
    H = int(TEMPLATE_WID_PX * scale) + 2 * margin
    canvas = np.full((H, W, 3), (40, 100, 40), np.uint8)   # pitch green

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return (int(p[0] * scale) + margin, int(p[1] * scale) + margin)

    verts_tpl = [np.array([cx / TEMPLATE_SCALE, cy / TEMPLATE_SCALE])
                 for cx, cy in CONFIG.vertices]
    for a, b in CONFIG.edges:
        cv2.line(canvas, to_px(verts_tpl[a - 1]), to_px(verts_tpl[b - 1]),
                 (210, 210, 210), 1, cv2.LINE_AA)
    # Centre circle (drawn explicitly; the vertices only mark intersections).
    centre = np.array([TEMPLATE_LEN_PX / 2, TEMPLATE_WID_PX / 2])
    cv2.circle(canvas, to_px(centre),
               int(CONFIG.centre_circle_radius / TEMPLATE_SCALE * scale),
               (210, 210, 210), 1, cv2.LINE_AA)
    return canvas, scale, margin


def viz_topdown(
    ground_points: list[PlayerGroundPoint],
    frame_grade: FrameGrade,
    homography: HomographyResult,
    scale: float = 1.0,
    margin: int = 20,
) -> np.ndarray:
    """
    Top-down scatter: players by TEAM_BGR, the trusted region (image hull mapped
    to pitch space) drawn, out-of-hull players flagged with an X, grade printed.
    """
    canvas, s, m = _draw_pitch_template(scale, margin)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return (int(p[0] * s) + m, int(p[1] * s) + m)

    # Trusted region: project the image-space inlier hull into pitch space.
    if homography.H is not None and len(frame_grade.hull_pts) >= 3:
        hull_tpl = cv2.perspectiveTransform(
            frame_grade.hull_pts.reshape(-1, 1, 2).astype(np.float32),
            homography.H).reshape(-1, 2)
        hull_px = np.array([to_px(p) for p in hull_tpl], np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [hull_px], (90, 150, 90))
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [hull_px], True, (230, 230, 120), 2, cv2.LINE_AA)

    for gp in ground_points:
        if gp.pitch_xy is None:
            continue
        c = to_px(gp.pitch_xy)
        color = TEAM_BGR[gp.team % len(TEAM_BGR)]
        cv2.circle(canvas, c, 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 8, (255, 255, 255), 1, cv2.LINE_AA)
        if gp.signed_dist_px > 0:    # outside hull → flag
            cv2.line(canvas, (c[0] - 7, c[1] - 7), (c[0] + 7, c[1] + 7),
                     (40, 40, 240), 2, cv2.LINE_AA)
            cv2.line(canvas, (c[0] - 7, c[1] + 7), (c[0] + 7, c[1] - 7),
                     (40, 40, 240), 2, cv2.LINE_AA)

    badge = {Grade.SOLID: (60, 220, 60), Grade.PARTIAL: (40, 200, 240),
             Grade.ABSTAIN: (40, 40, 240)}[frame_grade.grade]
    cv2.putText(canvas, f"GRADE: {frame_grade.grade.value.upper()}  "
                f"(inliers={frame_grade.n_inliers}, "
                f"reproj={frame_grade.mean_reproj_px:.1f}px)",
                (m, canvas.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                badge, 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyse(
    image_path: str,
    api_key: str | None = None,
    conf_threshold: float = 0.5,
    ransac_reproj_threshold: float = 8.0,
    out_prefix: str | None = None,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] cannot read image: {image_path!r}")
    print(f"[1/7] Image: {image_path}  ({image.shape[1]}x{image.shape[0]})")

    if out_prefix is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_prefix = base.replace(" ", "_")

    # Players + teams (reuse the existing detection / 3-feature team clustering).
    players, _ball, backend = detect_all(image, api_key, DEFAULT_MODEL_ID)
    print(f"[2/7] Players: {len(players)} (backend={backend})")
    team_labels = assign_teams_2d(players, image) if players else np.array([], int)

    # Pitch keypoints (isolated hosted-API parser).
    kps = detect_pitch_keypoints(image, api_key, conf_threshold=conf_threshold)
    print(f"[3/7] Confident keypoints (>= {conf_threshold}): {len(kps)}  "
          f"classes={sorted(kp.index + 1 for kp in kps)}")

    # Solve + grade (pure functions of correspondences).
    homo = solve_homography(kps, ransac_reproj_threshold)
    print(f"[4/7] Homography: "
          f"{'OK' if homo.H is not None else 'FAILED'}  "
          f"inliers={homo.n_inliers}  reproj={homo.mean_reproj_px:.2f}px  "
          f"centre-circle inliers={homo.centre_circle_inliers}")

    grade = grade_frame(homo, image.shape)
    print(f"[5/7] Frame grade: {grade.grade.value.upper()}  "
          f"(hull covers {grade.hull_area_frac * 100:.1f}% of image)")

    # Map players + pairwise confidence.
    gps = map_players(players, team_labels, homo, grade)
    pairs = pairwise_confidence(gps, grade)
    n_in = sum(1 for g in gps if g.signed_dist_px <= 0)
    print(f"[6/7] Players mapped: {len(gps)}  inside-hull={n_in}  "
          f"out-of-hull={len(gps) - n_in}")
    if pairs:
        confs = np.array([p.conf for p in pairs])
        print(f"      Pairwise relations: {len(pairs)}  "
              f"conf min/mean/max = {confs.min():.2f}/{confs.mean():.2f}/{confs.max():.2f}  "
              f"(>0.5: {int((confs > 0.5).sum())})")

    # Debug viz.
    kp_path = f"{out_prefix}_keypoints.png"
    td_path = f"{out_prefix}_topdown.png"
    cv2.imwrite(kp_path, viz_keypoints_on_frame(image, kps, homo))
    cv2.imwrite(td_path, viz_topdown(gps, grade, homo))
    print(f"[7/7] Saved → {kp_path}, {td_path}")

    if grade.grade == Grade.ABSTAIN:
        print("\n[ABSTAIN] Geometry insufficient — no confident relations emitted. "
              "(This is a correct output, not a failure.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pitch homography + relative-confidence layer.")
    ap.add_argument("image", help="Path to input football still")
    ap.add_argument("--api-key", default=None, help="Roboflow API key (or ROBOFLOW_API_KEY)")
    ap.add_argument("--conf", type=float, default=0.5, help="Keypoint confidence threshold")
    ap.add_argument("--ransac", type=float, default=8.0,
                    help="RANSAC reprojection threshold (template px)")
    ap.add_argument("-o", "--out-prefix", default=None,
                    help="Output filename prefix (default: derived from image name)")
    args = ap.parse_args()
    analyse(args.image, args.api_key, args.conf, args.ransac, args.out_prefix)


if __name__ == "__main__":
    main()
