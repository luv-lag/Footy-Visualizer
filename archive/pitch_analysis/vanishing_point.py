#!/usr/bin/env python3
"""
Vanishing Point Estimator for Football Match Images
====================================================
Uses player depth-line geometry to estimate the pitch vanishing point.

Usage:
    ROBOFLOW_API_KEY=<key> python vanishing_point.py match.jpg
    python vanishing_point.py match.jpg --api-key <key>
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings

import numpy as np
import cv2

from football_analysis import detect_all, DEFAULT_MODEL_ID, TEAM_BGR

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

OUTPUT_PATH = "vp_output.png"
NODE_RADIUS = 10


# ── team separation ───────────────────────────────────────────────────────────

def _jersey_hsv(image_bgr: np.ndarray, bbox: list[int]) -> tuple[float, float, float]:
    """
    Saturation-weighted mean hue, mean saturation, and mean value from the torso crop.
    All three are normalised to [0, 1] so KMeans weights them equally.
    """
    x1, y1, x2, y2 = bbox
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    r1 = y1 + max(1, int(h * 0.15))
    r2 = y1 + max(2, int(h * 0.55))
    c1 = x1 + max(1, int(w * 0.20))
    c2 = x2 - max(1, int(w * 0.20))
    crop = image_bgr[r1:r2, c1:c2]
    if crop.size == 0:
        return 0.5, 0.5, 0.5
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue_weighted = float((hue * sat).sum() / (sat.sum() + 1e-6))
    return hue_weighted / 180.0, float(sat.mean()) / 255.0, float(val.mean()) / 255.0


def assign_teams_2d(players: list[dict], image_bgr: np.ndarray) -> np.ndarray:
    """KMeans(k=2) on [hue, saturation, value] triples extracted from each player's torso."""
    if len(players) < 2:
        return np.zeros(len(players), dtype=int)
    features = np.array(
        [_jersey_hsv(image_bgr, p["bbox"]) for p in players], dtype=float
    )
    km = KMeans(n_clusters=min(2, len(players)), random_state=42, n_init=10)
    return km.fit_predict(features).astype(int)


# ── geometry helpers ──────────────────────────────────────────────────────────

def _center(player: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = player["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _cluster_channels(
    players: list[dict],
    indices: np.ndarray,
    x_threshold: float,
) -> list[list[int]]:
    """
    Sort by X, group consecutive players within x_threshold into channels.
    Returns list of channels; each channel is a list of global player indices.
    """
    if len(indices) == 0:
        return []
    cx_of = {int(i): _center(players[i])[0] for i in indices}
    sorted_idx = sorted(int(i) for i in indices)
    sorted_idx.sort(key=lambda i: cx_of[i])

    channels: list[list[int]] = [[sorted_idx[0]]]
    channel_means: list[float] = [cx_of[sorted_idx[0]]]
    for k in range(1, len(sorted_idx)):
        curr_i = sorted_idx[k]
        if abs(cx_of[curr_i] - channel_means[-1]) <= x_threshold:
            channels[-1].append(curr_i)
            channel_means[-1] = float(np.mean([cx_of[j] for j in channels[-1]]))
        else:
            channels.append([curr_i])
            channel_means.append(cx_of[curr_i])
    return channels


def _fit_depth_line(
    players: list[dict],
    ch: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Orthogonal-regression line via SVD on mean-centred points.
    Returns (centroid, direction, mean_orthogonal_residual).
    """
    pts = np.array([_center(players[i]) for i in ch], dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]                              # first principal component

    proj = centered @ direction
    residuals = np.linalg.norm(centered - np.outer(proj, direction), axis=1)
    return centroid, direction, float(residuals.mean())


def _line_intersection(
    c1: np.ndarray, d1: np.ndarray,
    c2: np.ndarray, d2: np.ndarray,
) -> np.ndarray | None:
    """Intersect two 2-D lines (point + direction). Returns None if parallel."""
    cos_a = abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
    angle = min(np.degrees(np.arccos(np.clip(cos_a, 0.0, 1.0))), 180.0)
    if min(angle, 180.0 - angle) < 1.0:
        return None
    A = np.column_stack([d1, -d2])
    b = c2 - c1
    try:
        ts = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return c1 + ts[0] * d1


def _extend_to_bounds(
    cx: float, cy: float,
    dx: float, dy: float,
    W: int, H: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return two endpoints of the line clipped to image bounds."""
    ts: list[float] = []
    eps = 1e-9
    if abs(dx) > eps:
        for xe in (0.0, float(W - 1)):
            t = (xe - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= H - 1:
                ts.append(t)
    if abs(dy) > eps:
        for ye in (0.0, float(H - 1)):
            t = (ye - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= W - 1:
                ts.append(t)
    if len(ts) < 2:
        return None
    t_min, t_max = min(ts), max(ts)
    if abs(t_max - t_min) < eps:
        return None
    p1 = (int(round(cx + t_min * dx)), int(round(cy + t_min * dy)))
    p2 = (int(round(cx + t_max * dx)), int(round(cy + t_max * dy)))
    return p1, p2


def _draw_star(
    img: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled 5-pointed star."""
    cx, cy = center
    pts = []
    for k in range(10):
        angle = np.radians(-90.0 + k * 36.0)
        r = size if k % 2 == 0 else size // 2
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [arr], color)
    cv2.polylines(img, [arr], True, (0, 0, 0), 1, cv2.LINE_AA)


# ── main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Football vanishing point estimator.")
    ap.add_argument("image", help="Path to input football image")
    ap.add_argument("--api-key", default=None,
                    help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"[ERROR] Cannot read image: {args.image!r}")
    H, W = image.shape[:2]

    api_key = (args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")).strip() or None

    # Step 1 – detection + team separation
    print("[1/7] Detecting players …")
    players, _ball, backend = detect_all(image, api_key, args.model_id)
    print(f"      backend={backend}  players={len(players)}")

    if len(players) < 2:
        sys.exit("[ERROR] Fewer than 2 players detected.")

    print("[2/7] Assigning teams …")
    team_labels = assign_teams_2d(players, image)
    for t in range(2):
        print(f"      Team {t + 1}: {int((team_labels == t).sum())} players")

    # Step 2 – x_threshold from all players, then cluster per team
    x_threshold = W * 0.06
    print(f"[3/7] x_threshold = {x_threshold:.1f} px  (image width = {W} px)")

    raw_channels: dict[int, list[list[int]]] = {}
    for t in range(2):
        indices = np.where(team_labels == t)[0]
        raw_channels[t] = _cluster_channels(players, indices, x_threshold)
        sizes = [len(ch) for ch in raw_channels[t]]
        print(f"      Team {t + 1}: {len(raw_channels[t])} channel(s) → sizes {sizes}")

    # Step 3 – discard channels with < 3 players
    valid_pre: list[tuple[int, list[int]]] = []
    for t in range(2):
        for ch in raw_channels[t]:
            if len(ch) >= 3:
                valid_pre.append((t, ch))

    total_valid_players = sum(len(ch) for _, ch in valid_pre)
    if total_valid_players < 6:
        print(f"[WARN] Only {total_valid_players} valid players across all channels "
              f"(need ≥ 6). Exiting.")
        sys.exit(0)

    print(f"[4/7] Channels with ≥ 3 players: {len(valid_pre)}")

    # Step 4 – fit depth lines; dynamic residual threshold = mean_residual * 1.5
    all_fitted: list[tuple[int, list[int], np.ndarray, np.ndarray, float]] = []
    for team_id, ch in valid_pre:
        centroid, direction, mean_res = _fit_depth_line(players, ch)
        all_fitted.append((team_id, ch, centroid, direction, mean_res))

    residual_limit = float(np.mean([r for *_, r in all_fitted])) * 1.5
    print(f"[4/7] Dynamic residual limit: {residual_limit:.1f} px  "
          f"(mean={residual_limit / 1.5:.1f} px × 1.5)")

    fitted: list[tuple[int, list[int], np.ndarray, np.ndarray]] = []
    channels_passing: dict[int, int] = {0: 0, 1: 0}

    for team_id, ch, centroid, direction, mean_res in all_fitted:
        if mean_res <= residual_limit:
            fitted.append((team_id, ch, centroid, direction))
            channels_passing[team_id] += 1
        else:
            print(f"      Team {team_id + 1} channel ({len(ch)} players) discarded: "
                  f"mean_residual={mean_res:.1f} px > {residual_limit:.1f} px")

    print(f"      Lines passing residual filter: {len(fitted)}")

    if len(fitted) < 2:
        print("[WARN] Fewer than 2 valid lines – cannot triangulate vanishing point.")
        sys.exit(0)

    # Step 5 – pairwise intersections → vanishing point
    intersections: list[np.ndarray] = []
    for i in range(len(fitted)):
        for j in range(i + 1, len(fitted)):
            _, _, c1, d1 = fitted[i]
            _, _, c2, d2 = fitted[j]
            pt = _line_intersection(c1, d1, c2, d2)
            if pt is not None:
                intersections.append(pt)

    if not intersections:
        print("[WARN] All line pairs were parallel – cannot compute vanishing point.")
        sys.exit(0)

    ipts = np.array(intersections)
    vp_x = float(np.median(ipts[:, 0]))
    vp_y = float(np.median(ipts[:, 1]))
    vp = (int(round(vp_x)), int(round(vp_y)))

    # Step 6 – draw output
    print("[6/7] Rendering …")
    out = image.copy()

    # Depth lines per team in team colour
    for team_id, _ch, centroid, direction in fitted:
        color = TEAM_BGR[team_id % len(TEAM_BGR)]
        endpoints = _extend_to_bounds(
            centroid[0], centroid[1],
            direction[0], direction[1],
            W, H,
        )
        if endpoints:
            cv2.line(out, endpoints[0], endpoints[1], color, 2, cv2.LINE_AA)

    # Intra-channel edges (drawn before nodes so nodes sit on top)
    for team_id, ch, _centroid, _direction in fitted:
        color = TEAM_BGR[team_id % len(TEAM_BGR)]
        pts = [tuple(int(v) for v in _center(players[i])) for i in ch]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                cv2.line(out, pts[a], pts[b], color, 1, cv2.LINE_AA)

    # Player nodes coloured by team
    for i, p in enumerate(players):
        cx_p, cy_p = _center(p)
        c = (int(cx_p), int(cy_p))
        color = TEAM_BGR[int(team_labels[i]) % len(TEAM_BGR)]
        cv2.circle(out, c, NODE_RADIUS + 2, (0, 0, 0),     -1, cv2.LINE_AA)
        cv2.circle(out, c, NODE_RADIUS,     color,          -1, cv2.LINE_AA)
        cv2.circle(out, c, NODE_RADIUS,     (255, 255, 255),  1, cv2.LINE_AA)

    # White "true vertical" line from vanishing point to player centroid
    all_centers = np.array([_center(p) for p in players])
    pc = all_centers.mean(axis=0)
    pc_pt = (int(round(pc[0])), int(round(pc[1])))
    cv2.line(out, vp, pc_pt, (255, 255, 255), 2, cv2.LINE_AA)
    mid_x = (vp[0] + pc_pt[0]) // 2 + 5
    mid_y = (vp[1] + pc_pt[1]) // 2
    cv2.putText(out, "true vertical", (mid_x, mid_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Yellow star at vanishing point
    _draw_star(out, vp, 18, (0, 215, 255))

    cv2.imwrite(OUTPUT_PATH, out)
    print(f"[7/7] Saved → {OUTPUT_PATH}")

    # Step 7 – terminal summary
    print("\n=== Vanishing Point Analysis ===")
    for t in range(2):
        print(f"  Team {t + 1} players detected   : {int((team_labels == t).sum())}")
    for t in range(2):
        print(f"  Team {t + 1} channels found     : {len(raw_channels[t])}")
    for t in range(2):
        print(f"  Team {t + 1} channels (passing) : {channels_passing[t]}")
    print(f"  Vanishing point            : ({vp[0]}, {vp[1]})")
    print(f"  VP distance from top       : {vp[1]} px")


if __name__ == "__main__":
    main()
