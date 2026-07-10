#!/usr/bin/env python3
"""
Football Tactical Analysis Tool
================================
Pipeline
--------
  1. Read a static match image
  2. Detect players + ball via Roboflow YOLOv8 model (direct REST API call)
     Falls back to YOLOv8n-COCO person/sports-ball when no API key is set.
  3. Separate teams by jersey colour – KMeans(k=2) on saturation-weighted hue
  4. Identify formation lines per team – DBSCAN on player X coordinates (broadcast camera: lines run left-right)
  5. Count players per line → formation string (e.g. "4-3-3")
  6. Draw overlay: coloured nodes, intra-line edges, ball ring → output.png

Usage
-----
    # Recommended – uses the football-specific Roboflow model:
    ROBOFLOW_API_KEY=<key> python football_analysis.py match.jpg

    python football_analysis.py match.jpg --api-key <key>
    python football_analysis.py match.jpg -o annotated.png

    # No key → YOLOv8n-COCO fallback (detects generic persons + sports ball):
    python football_analysis.py match.jpg

Get a free Roboflow API key at https://app.roboflow.com (Settings → API).

Default Roboflow model
----------------------
  workspace : roboflow-jvuqo
  project   : football-players-detection-3zvbc
  version   : 1
  classes   : player, goalkeeper, referee, ball

Override with:  --model-id <project>/<version>
"""

from __future__ import annotations
import base64
import os
import sys
import argparse
import warnings

import numpy as np
import cv2
import requests
from sklearn.cluster import KMeans, DBSCAN

warnings.filterwarnings("ignore")


# ─── visual constants ──────────────────────────────────────────────────────────
TEAM_BGR: list[tuple[int, int, int]] = [
    (50,  80, 230),    # Team A – red
    (230, 120,  50),   # Team B – blue
]
BALL_BGR    = (0, 230, 230)    # cyan-yellow
NODE_RADIUS = 12
EDGE_TH     = 2
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE  = 0.65
FONT_TH     = 2

DEFAULT_MODEL_ID       = "football-players-detection-3zvbc/1"
ROBOFLOW_INFER_URL     = "https://detect.roboflow.com"
ROBOFLOW_INFER_TIMEOUT = 30  # seconds

# Classes the Roboflow football model returns that are actual field players.
# "referee" and "linesman" (and any future non-player class) are intentionally
# excluded so they don't inflate team counts or distort formation detection.
PLAYER_CLASSES = {"player", "goalkeeper"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _encode_image(image_bgr: np.ndarray) -> str:
    """JPEG-encode image and return as base64 string."""
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def _roboflow_infer(image_bgr: np.ndarray, model_id: str, api_key: str) -> dict:
    """
    POST to the Roboflow hosted inference REST API.
    model_id format: "<project>/<version>" or "<workspace>/<project>/<version>"
    """
    parts = model_id.split("/")
    if len(parts) == 2:
        endpoint = f"{ROBOFLOW_INFER_URL}/{parts[0]}/{parts[1]}"
    elif len(parts) == 3:
        endpoint = f"{ROBOFLOW_INFER_URL}/{parts[1]}/{parts[2]}"
    else:
        raise ValueError(f"Unexpected model_id format: {model_id!r}  (expect project/version)")

    img_b64 = _encode_image(image_bgr)
    resp = requests.post(
        endpoint,
        params={"api_key": api_key},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=ROBOFLOW_INFER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_roboflow_response(data: dict):
    """
    Parse Roboflow inference JSON into (players, ball).

    players : list[dict]  {"bbox": [x1,y1,x2,y2], "conf": float, "label": str}
    ball    : dict | None {"bbox": ..., "conf": float}
    """
    players: list[dict] = []
    ball: dict | None = None

    img_w = data.get("image", {}).get("width", 0)
    img_h = data.get("image", {}).get("height", 0)

    # Diagnostic: print every unique class the model returned.
    raw_classes = sorted({p.get("class", "") for p in data.get("predictions", [])})
    print(f"      [diag] Roboflow classes in response: {raw_classes}")

    for pred in data.get("predictions", []):
        cx, cy = pred["x"], pred["y"]
        w,  h  = pred["width"], pred["height"]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = int(cx + w / 2) if not img_w else min(img_w, int(cx + w / 2))
        y2 = int(cy + h / 2) if not img_h else min(img_h, int(cy + h / 2))
        bbox = [x1, y1, x2, y2]
        cls  = pred.get("class", "").lower()
        conf = float(pred.get("confidence", 0.0))

        if "ball" in cls:
            if ball is None or conf > ball["conf"]:
                ball = {"bbox": bbox, "conf": conf}
        elif cls in PLAYER_CLASSES:
            # Only accept known player classes; referee/linesman etc. are skipped.
            players.append({"bbox": bbox, "conf": conf, "label": cls})
        else:
            print(f"      [diag] Skipping non-player detection: class={cls!r} conf={conf:.2f}")

    return players, ball


def _detect_ultralytics(image_bgr: np.ndarray):
    """
    Fallback: detect with a local YOLOv8n-COCO model.
    Classes used: person → player,  sports ball → ball.
    """
    from ultralytics import YOLO  # type: ignore[import]
    model = YOLO("yolov8n.pt")
    results = model(image_bgr, verbose=False)[0]
    players: list[dict] = []
    ball: dict | None = None

    for box in results.boxes:
        label = model.names[int(box.cls)].lower()
        conf  = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]
        if "ball" in label:
            if ball is None or conf > ball["conf"]:
                ball = {"bbox": bbox, "conf": conf}
        elif label == "person":
            players.append({"bbox": bbox, "conf": conf, "label": "player"})

    return players, ball


def detect_all(image_bgr: np.ndarray, api_key: str | None, model_id: str):
    """
    Top-level detection dispatcher.
    Returns (players, ball, backend_name).
    """
    key = (api_key or os.environ.get("ROBOFLOW_API_KEY", "")).strip()

    if key:
        print(f"      Calling Roboflow API  ({model_id}) …")
        try:
            data = _roboflow_infer(image_bgr, model_id, key)
            players, ball = _parse_roboflow_response(data)
            return players, ball, "roboflow"
        except Exception as exc:
            print(f"[WARN] Roboflow API failed ({exc}); falling back to YOLOv8n-COCO.")

    print("      Using local YOLOv8n-COCO (no Roboflow API key set) …")
    players, ball = _detect_ultralytics(image_bgr)
    return players, ball, "ultralytics"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – TEAM CLUSTERING BY JERSEY COLOUR
# ══════════════════════════════════════════════════════════════════════════════

def _jersey_hue(image_bgr: np.ndarray, bbox: list[int]) -> float:
    """
    Dominant hue of the torso crop, weighted by pixel saturation.
    Returns a value in [0, 179] (OpenCV hue scale).
    """
    x1, y1, x2, y2 = bbox
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    # Torso band: rows 15-55 %, cols 20-80 %
    r1 = y1 + max(1, int(h * 0.15))
    r2 = y1 + max(2, int(h * 0.55))
    c1 = x1 + max(1, int(w * 0.20))
    c2 = x2 - max(1, int(w * 0.20))
    crop = image_bgr[r1:r2, c1:c2]
    if crop.size == 0:
        return 90.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    return float((hue * sat).sum() / (sat.sum() + 1e-6))


def assign_teams(players: list[dict], image_bgr: np.ndarray) -> np.ndarray:
    """
    KMeans(k=2) on jersey hue.
    Returns int array shape (n_players,) with values 0 or 1.
    """
    if len(players) < 2:
        return np.zeros(len(players), dtype=int)
    hues = np.array([_jersey_hue(image_bgr, p["bbox"]) for p in players]).reshape(-1, 1)
    km   = KMeans(n_clusters=min(2, len(players)), random_state=42, n_init=10)
    return km.fit_predict(hues).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – FORMATION DETECTION VIA DBSCAN
# ══════════════════════════════════════════════════════════════════════════════

def _cx(player: dict) -> float:
    x1, y1, x2, y2 = player["bbox"]
    return (x1 + x2) / 2.0


def detect_formation(
    players     : list[dict],
    team_labels : np.ndarray,
    team_id     : int,
) -> tuple[np.ndarray, str, np.ndarray, list[int]]:
    """
    Cluster one team's players into formation lines using DBSCAN on X coords.

    In a broadcast side-on camera the pitch runs left-to-right in the image.
    Players in the same formation line (e.g. all four defenders) sit at
    roughly the same horizontal position (X), not the same vertical position.
    Clustering on X therefore recovers depth-from-goal lines correctly.

    Returns
    -------
    line_of        – int array shape (n_team,); line index 0 = left of image
    formation_str  – e.g. "4-3-3"
    team_indices   – global indices into `players` for this team
    sorted_line_ids– list of line ids ordered left → right
    """
    team_indices = np.where(team_labels == team_id)[0]
    n = len(team_indices)
    if n == 0:
        return np.array([], int), "", np.array([], int), []

    xs = np.array([_cx(players[i]) for i in team_indices]).reshape(-1, 1)

    # Adaptive eps ≈ 12 % of horizontal spread, minimum 30 px
    spread = float(xs.max() - xs.min()) if n > 1 else 1.0
    eps    = max(30.0, spread * 0.12)

    raw_labels = DBSCAN(eps=eps, min_samples=1).fit_predict(xs)

    # Re-assign noise points (-1) to the nearest cluster
    clusters = [c for c in np.unique(raw_labels) if c != -1]
    if not clusters:
        clusters   = [0]
        raw_labels = np.zeros(n, dtype=int)
    else:
        for idx in np.where(raw_labels == -1)[0]:
            dists      = {c: abs(xs[idx, 0] - xs[raw_labels == c, 0].mean()) for c in clusters}
            raw_labels[idx] = min(dists, key=dists.get)

    # Sort clusters left → right (ascending mean X)
    mean_x       = {c: xs[raw_labels == c, 0].mean() for c in clusters}
    sorted_clust = sorted(clusters, key=lambda c: mean_x[c])

    # Remap to consecutive 0, 1, 2, …
    remap   = {old: new for new, old in enumerate(sorted_clust)}
    line_of = np.array([remap[l] for l in raw_labels], dtype=int)
    sorted_line_ids = list(range(len(sorted_clust)))

    counts = [int((line_of == l).sum()) for l in sorted_line_ids]

    # Normalise so the formation always reads GK → forwards.
    # A team whose GK defends the right goal has its 1-player line at the
    # highest X (rightmost column); in that case reverse so GK comes first.
    if len(counts) >= 2 and counts[-1] == 1 and counts[0] != 1:
        counts = counts[::-1]

    formation_str = "-".join(str(c) for c in counts)

    return line_of, formation_str, team_indices, sorted_line_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def _cpt(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def draw_overlay(
    image       : np.ndarray,
    players     : list[dict],
    team_labels : np.ndarray,
    formations  : dict[int, tuple],
    ball        : dict | None,
) -> np.ndarray:
    out = image.copy()

    # ── intra-line edges (drawn first so nodes appear on top) ──────────────
    for team_id, (line_of, _fmt, team_indices, sorted_lines) in formations.items():
        color = TEAM_BGR[team_id % len(TEAM_BGR)]
        for line_id in sorted_lines:
            local_idx  = np.where(line_of == line_id)[0]
            global_idx = team_indices[local_idx]
            pts = [_cpt(players[i]["bbox"]) for i in global_idx]
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    cv2.line(out, pts[a], pts[b], color, EDGE_TH, cv2.LINE_AA)

    # ── player nodes ───────────────────────────────────────────────────────
    for i, p in enumerate(players):
        c     = _cpt(p["bbox"])
        color = TEAM_BGR[int(team_labels[i]) % len(TEAM_BGR)]
        cv2.circle(out, c, NODE_RADIUS + 2, (0, 0, 0),          -1, cv2.LINE_AA)  # drop shadow
        cv2.circle(out, c, NODE_RADIUS,     color,               -1, cv2.LINE_AA)  # fill
        cv2.circle(out, c, NODE_RADIUS,     (255, 255, 255),      1, cv2.LINE_AA)  # rim

    # ── ball ───────────────────────────────────────────────────────────────
    if ball is not None:
        x1, y1, x2, y2 = ball["bbox"]
        bc   = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        brad = max(10, int(((x2 - x1) + (y2 - y1)) / 4))
        cv2.circle(out, bc, brad + 3, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.circle(out, bc, brad,     BALL_BGR,     2, cv2.LINE_AA)
        cv2.circle(out, bc, 4,        BALL_BGR,    -1, cv2.LINE_AA)

    # ── formation legend bar (appended below the image) ────────────────────
    panel_h = 44
    panel   = np.zeros((panel_h, out.shape[1], 3), dtype=np.uint8)
    x_pos   = 14
    for team_id, (_lo, formation_str, _ti, _sl) in formations.items():
        color = TEAM_BGR[team_id % len(TEAM_BGR)]
        text  = f"Team {team_id + 1}:  {formation_str}"
        (tw, _th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_TH)
        cv2.putText(panel, text, (x_pos, panel_h - 12),
                    FONT, FONT_SCALE, color, FONT_TH, cv2.LINE_AA)
        x_pos += tw + 50

    return np.vstack([out, panel])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyse(
    image_path : str,
    output_path: str  = "output.png",
    api_key    : str | None = None,
    model_id   : str  = DEFAULT_MODEL_ID,
) -> None:

    # 1 ── load image ────────────────────────────────────────────────────────
    print(f"[1/5] Reading image       : {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] Cannot read image: '{image_path}'")

    # 2 ── detection ─────────────────────────────────────────────────────────
    print("[2/5] Running detection   …")
    players, ball, backend = detect_all(image, api_key, model_id)
    print(f"      backend={backend}  players={len(players)}  ball={'yes' if ball else 'no'}")

    if len(players) < 4:
        print("[WARN] Fewer than 4 players detected – writing image without analysis.")
        cv2.imwrite(output_path, image)
        return

    # 3 ── team clustering ───────────────────────────────────────────────────
    print("[3/5] Clustering teams    …")
    team_labels = assign_teams(players, image)
    for t in range(2):
        print(f"      Team {t + 1}: {(team_labels == t).sum()} players")

    # 4 ── formation detection ───────────────────────────────────────────────
    print("[4/5] Detecting formations…")
    formations: dict[int, tuple] = {}
    for team_id in range(2):
        line_of, formation_str, team_indices, sorted_lines = detect_formation(
            players, team_labels, team_id
        )
        if len(team_indices):
            formations[team_id] = (line_of, formation_str, team_indices, sorted_lines)
            print(f"      Team {team_id + 1} → {formation_str}")

    # 5 ── draw & save ───────────────────────────────────────────────────────
    print("[5/5] Drawing overlay     …")
    result = draw_overlay(image, players, team_labels, formations, ball)
    cv2.imwrite(output_path, result)
    print(f"\n[DONE] Annotated image saved → {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Football Tactical Analysis – formation detection + visual overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("image",
                    help="Path to the input football match image")
    ap.add_argument("-o", "--output", default="output.png",
                    help="Output path  (default: output.png)")
    ap.add_argument("--api-key", default=None,
                    help="Roboflow API key  (or set ROBOFLOW_API_KEY env var)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"Roboflow model id  (default: {DEFAULT_MODEL_ID})")
    args = ap.parse_args()
    analyse(args.image, args.output, args.api_key, args.model_id)


if __name__ == "__main__":
    main()
