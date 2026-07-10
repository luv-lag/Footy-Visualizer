#!/usr/bin/env python3
"""
Entry point: python orientation_test/main.py <image> [--eps N] [--min-samples N] [--debug]
"""
import argparse
import sys

import cv2
import numpy as np

from detect import detect_players
from cluster import run_dbscan, analyze_clusters, weighted_average_angle, rotate_coords
from formation import recover_formation
from draw import draw_output, draw_debug_schematic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover pitch orientation from a static football broadcast image."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--eps", type=float, default=80.0,
                        help="DBSCAN neighbourhood radius in pixels (default: 80)")
    parser.add_argument("--min-samples", type=int, default=4,
                        help="DBSCAN minimum cluster size (default: 4)")
    parser.add_argument("--debug", action="store_true",
                        help="Draw coordinate-axes overlay and write orientation_debug.png")
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: cannot load image '{args.image}'")

    # ── 1. Detect players ────────────────────────────────────────────────────
    print("Calling Roboflow API…")
    player_positions = detect_players(args.image)
    print(f"Detected players : {len(player_positions)}")

    coords = np.array(player_positions, dtype=float)

    # ── 2. First DBSCAN on raw XY ────────────────────────────────────────────
    labels = run_dbscan(coords, eps=args.eps, min_samples=args.min_samples)
    n_clusters = len(set(labels) - {-1})
    print(f"Clusters found   : {n_clusters}")

    # ── 3. PCA per cluster, aspect-ratio filter ──────────────────────────────
    clusters = analyze_clusters(labels, coords, min_aspect_ratio=2.5)
    passing = [c for c in clusters if c["passes"]]
    print(f"Clusters passing aspect-ratio filter (≥ 2.5): {len(passing)}")
    for c in passing:
        print(
            f"  cluster {c['label']:2d}  angle={c['angle']:+7.2f}°  "
            f"AR={c['aspect_ratio']:5.2f}  n={c['count']}"
        )

    # ── 4. Weighted average orientation ──────────────────────────────────────
    avg_angle = weighted_average_angle(clusters)
    print(f"Weighted avg orientation : {avg_angle:+.2f}°")

    # ── 5. Rotate coords and recover formation ───────────────────────────────
    rotated = rotate_coords(coords, avg_angle)
    formation, lines = recover_formation(rotated, eps=args.eps)
    print(f"Recovered formation : {formation}")

    # ── 6. Draw and save ─────────────────────────────────────────────────────
    import pathlib
    out_dir = pathlib.Path("Orientation Test Outputs")
    out_dir.mkdir(exist_ok=True)
    stem = pathlib.Path(args.image).stem.replace(" ", "_")
    out_path   = str(out_dir / f"{stem}_output.png")
    debug_path = str(out_dir / f"{stem}_debug.png")

    output = draw_output(img, coords, labels, clusters, avg_angle, formation, debug=args.debug)
    cv2.imwrite(out_path, output)
    print(f"Saved : {out_path}")

    if args.debug:
        schematic = draw_debug_schematic(coords, labels, rotated, avg_angle, img.shape)
        cv2.imwrite(debug_path, schematic)
        print(f"Saved : {debug_path}")


if __name__ == "__main__":
    main()
