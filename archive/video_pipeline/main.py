"""Entry point and argument parser for observer_recovery pipeline."""

import argparse
import os
import sys

import cv2
import numpy as np

from detection import detect_players
from tracking import build_tracks, stable_point_pairs
from homography import (
    compute_homography,
    estimate_focal_length,
    build_K,
    decompose_homography,
    refine_pose,
    recover_camera_params,
    estimate_pitch_width_px,
    _triangulate_pts3d,
)
from formation import detect_formations
from overlay import annotate_frame, open_writer, write_frame, LEGEND_HEIGHT

KEYFRAME_INTERVAL = 60


def extract_keyframes(cap: cv2.VideoCapture) -> list[tuple[int, np.ndarray]]:
    """Extract frames spaced KEYFRAME_INTERVAL apart across full video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        keyframes.append((idx, frame))
        idx += KEYFRAME_INTERVAL
    return keyframes


def build_warp_matrix(theta_deg: float, d_m: float, f: float, w: int, h: int) -> np.ndarray:
    """Construct perspective warp matrix from camera parameters.

    Maps original frame to approximate top-down view based on recovered
    elevation angle theta and focal length f.
    """
    theta = np.radians(theta_deg)
    # Four source points: corners of a rectangle on the pitch plane
    # estimated by projecting ground plane corners
    cy = h / 2.0
    cx = w / 2.0
    # Compute vertical stretch factor from elevation
    stretch = 1.0 / max(np.sin(theta), 0.1)

    src = np.float32([
        [0, h * 0.3],
        [w, h * 0.3],
        [w, h * 0.95],
        [0, h * 0.95],
    ])
    dst = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return M


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        sys.exit("Error: ROBOFLOW_API_KEY environment variable not set.")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} fps")
    print("Extracting keyframes...")
    keyframes = extract_keyframes(cap)
    print(f"  {len(keyframes)} keyframes extracted")

    # --- Detection ---
    print("Running player detection on keyframes...")
    all_detections: list[list[dict]] = []
    for fi, (frame_idx, frame) in enumerate(keyframes):
        print(f"  Keyframe {fi+1}/{len(keyframes)} (frame {frame_idx})", end="\r")
        dets = detect_players(frame, api_key)
        all_detections.append(dets)
    print()

    # --- Tracking ---
    print("Building stable tracks...")
    tracks = build_tracks(all_detections)
    print(f"  {len(tracks)} stable tracks found")

    # --- Per-keyframe-pair homography, pose recovery ---
    n_kf = len(keyframes)
    pair_results = []  # list of (d_m, theta_deg, reproj_err, f)

    for i in range(n_kf - 1):
        pts_a, pts_b = stable_point_pairs(tracks, i, i + 1)
        if len(pts_a) < 4:
            continue

        H = compute_homography(pts_a, pts_b)
        if H is None:
            continue

        _, frame_a = keyframes[i]
        f = estimate_focal_length(H, frame_w)
        K = build_K(f, frame_w, frame_h)

        result = decompose_homography(H, K, pts_a, pts_b)
        if result is None:
            continue
        R, t, reproj_err = result

        # LM refinement
        pts3d, inlier_mask = _triangulate_pts3d(R, t, pts_a, pts_b, K)
        R, t = refine_pose(R, t, pts_b, pts3d, K, inlier_mask)

        pitch_w_px = estimate_pitch_width_px(pts_a)
        d_m, theta_deg = recover_camera_params(R, t, pts_a, pitch_w_px, f)

        pair_results.append((d_m, theta_deg, reproj_err, f))
        print(
            f"  Pair {i}-{i+1}: d={d_m:.1f}m  theta={theta_deg:.1f}°  "
            f"reproj={reproj_err:.1f}px  f={f:.0f}px"
        )

    if not pair_results:
        sys.exit("No valid keyframe pairs could be processed. Exiting.")

    # Weighted average (weight = 1/reproj_err)
    ds = np.array([r[0] for r in pair_results])
    thetas = np.array([r[1] for r in pair_results])
    errs = np.array([r[2] for r in pair_results])
    fs = np.array([r[3] for r in pair_results])

    weights = 1.0 / np.maximum(errs, 1e-6)
    weights /= weights.sum()

    final_d = float(np.dot(weights, ds))
    final_theta = float(np.dot(weights, thetas))
    final_f = float(np.dot(weights, fs))
    final_reproj = float(np.mean(errs))

    print(f"\nFinal camera parameters:")
    print(f"  Elevation angle : {final_theta:.2f} deg")
    print(f"  Observer distance: {final_d:.2f} m")
    print(f"  Focal length    : {final_f:.1f} px")
    print(f"  Mean reproj err : {final_reproj:.2f} px")

    # --- Build warp matrix from final params ---
    warp_M = build_warp_matrix(final_theta, final_d, final_f, frame_w, frame_h)
    inv_warp = np.linalg.inv(warp_M)

    # --- Process all frames ---
    print(f"\nWriting annotated video to {args.output}...")
    writer = open_writer(args.output, fps, frame_w, frame_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Build a simple frame->detection lookup from keyframes
    kf_lookup = {fi: dets for (fi, _), dets in zip(keyframes, all_detections)}
    kf_indices = sorted(kf_lookup.keys())
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    last_result: dict = {
        "team_a_formation": "",
        "team_b_formation": "",
        "team_a_players": [],
        "team_b_players": [],
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use nearest keyframe detections for formation
        nearest_kf = min(kf_indices, key=lambda k: abs(k - frame_idx))
        dets = kf_lookup.get(nearest_kf, [])

        if dets:
            result = detect_formations(dets, warp_M, inv_warp, frame_w, frame_h)
            last_result = result
        else:
            result = last_result

        # Find ball in detections (class not in PLAYER_CLASSES)
        ball = None
        all_kf_dets_raw = kf_lookup.get(nearest_kf, [])

        annotated = annotate_frame(frame, result, ball=ball)
        write_frame(writer, annotated)

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}", end="\r")
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"\nDone. Output written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover camera position from football player tracking and output formation-annotated video."
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument(
        "-o", "--output", default="output.mp4", help="Output video path (default: output.mp4)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
