"""
Approach 4 — player detection module.

Wraps the Roboflow-YOLO detection + KMeans jersey-hue team assignment from
pitch_analysis/football_analysis.py unchanged, and adds save_detections()
to persist the resulting bbox/foot-point/team data as a JSON artefact.

Usage (CLI):
    python approach_4/player_detection.py <image> [--output-dir <dir>]
                                           [--api-key <key>]

Run from the football-analysis repo root.
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import cv2
import numpy as np

# Add repo root to path so pitch_analysis imports work regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pitch_analysis.football_analysis import (
    detect_all,
    assign_teams,
    DEFAULT_MODEL_ID,
)


def save_detections(
    players: list[dict],
    team_labels: np.ndarray,
    image_path: str,
    output_path: str,
) -> None:
    """
    Persist player detections as a JSON file.

    For each player the saved record contains:
      player_index  — position in the detection list
      bbox          — [x_min, y_min, x_max, y_max]  (image pixels)
      foot_point    — [(x_min+x_max)/2, y_max]       (ground-contact point)
      team          — KMeans jersey-hue label (0 or 1)
      conf          — detector confidence
      label         — class string ("player" / "goalkeeper")
    """
    records = []
    for i, (p, t) in enumerate(zip(players, team_labels)):
        x1, y1, x2, y2 = p["bbox"]
        records.append({
            "player_index": i,
            "bbox":         [x1, y1, x2, y2],
            "foot_point":   [(x1 + x2) / 2.0, float(y2)],
            "team":         int(t),
            "conf":         float(p.get("conf", p.get("confidence", 0.0))),
            "label":        p.get("label", "player"),
        })

    out = {
        "image":    image_path,
        "n_players": len(records),
        "team_counts": {
            "team_0": int((team_labels == 0).sum()),
            "team_1": int((team_labels == 1).sum()),
        },
        "players": records,
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


def run(
    image_path: str,
    output_dir: str | None = None,
    api_key: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[list[dict], np.ndarray, dict]:
    """
    Run detection + team assignment on one image.
    Returns (players, team_labels, summary_dict).
    """
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path!r}")

    players, ball, backend = detect_all(image, api_key, model_id)
    print(f"  backend={backend}  players={len(players)}  "
          f"ball={'yes' if ball else 'no'}")

    if len(players) < 2:
        print("[WARN] Fewer than 2 players detected — team assignment skipped.")
        team_labels = np.zeros(len(players), dtype=int)
    else:
        team_labels = assign_teams(players, image)

    for t in range(2):
        print(f"  team {t}: {int((team_labels == t).sum())} players")

    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))

    stem = os.path.splitext(os.path.basename(image_path))[0]
    det_path = os.path.join(output_dir, f"{stem}_detections.json")

    save_detections(players, team_labels, image_path, det_path)
    print(f"  Detections saved → {det_path}")

    summary = {
        "n_players":   len(players),
        "team_counts": {"team_0": int((team_labels == 0).sum()),
                        "team_1": int((team_labels == 1).sum())},
        "detections_path": det_path,
    }
    return players, team_labels, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect players and save structured detections JSON."
    )
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to write detections JSON (default: same as image)")
    ap.add_argument("--api-key", default=None,
                    help="Roboflow API key (or ROBOFLOW_API_KEY env var)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    key = (args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")) or None
    run(args.image, args.output_dir, key, args.model_id)


if __name__ == "__main__":
    main()
