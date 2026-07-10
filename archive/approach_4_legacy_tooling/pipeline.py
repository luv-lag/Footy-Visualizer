"""
pipeline.py — approach_4 PnLCalib player-positioning pipeline.

Orchestrates:
  1. PnLCalib calibration (calibration.py)
  2. Roboflow/YOLO player detection + KMeans team assignment (player_detection.py)
  3. Foot-pixel → metric-world projection (projection.py)
  4. Top-down pitch plot (visualization.py)
  5. JSON output (io_utils.py)

Usage (run from football-analysis/ repo root):
    /path/to/PnLCalib/.venv/bin/python approach_4/pipeline.py <image> \
        [--output-dir <dir>] [--api-key <key>] [--device cuda:0]

Requires: PnLCalib venv (torch, torchvision, etc.)
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── repo-root on sys.path so sibling packages resolve ────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── approach_4 modules ────────────────────────────────────────────────────────
from approach_4 import calibration as calib_mod
from approach_4.calibration  import CalibGrade, load_models, calibrate
from approach_4.projection   import project_players
from approach_4.visualization import plot_topdown
from approach_4.io_utils     import save_json, calib_result_to_dict

# player_detection.run() uses pitch_analysis imports; it handles its own
# sys.path setup, so we can call it directly.
from approach_4 import player_detection


# ── helpers ───────────────────────────────────────────────────────────────────

def _stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    image_path : str,
    output_dir : str | None = None,
    api_key    : str | None = None,
    device     : str = calib_mod.DEVICE,
) -> dict:
    """Run the full pipeline on one image.

    Returns a summary dict. All artefacts are written to output_dir.
    """
    t0 = time.time()

    # ── resolve paths ─────────────────────────────────────────────────────────
    image_path = os.path.abspath(image_path)
    if output_dir is None:
        output_dir = os.path.join(_REPO_ROOT, "outputs", "approach_4",
                                  _stem(image_path))
    os.makedirs(output_dir, exist_ok=True)

    stem = _stem(image_path)
    calib_json_path  = os.path.join(output_dir, f"{stem}_calib.json")
    proj_json_path   = os.path.join(output_dir, f"{stem}_projections.json")
    topdown_png_path = os.path.join(output_dir, f"{stem}_topdown.png")

    # ── 1. read image ─────────────────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path!r}")
    h_px, w_px = frame.shape[:2]
    print(f"[pipeline] image: {image_path}  ({w_px}×{h_px})")

    # ── 2. PnLCalib calibration ───────────────────────────────────────────────
    print("[pipeline] loading PnLCalib models …")
    model_kp, model_l = load_models(device=device)

    print("[pipeline] calibrating frame …")
    calib = calibrate(frame, model_kp, model_l, device=device)
    print(f"[pipeline] calibration grade={calib.grade.value}  "
          f"n_kp={calib.n_kp}  reproj={calib.reproj_err:.2f} px")

    save_json(calib_result_to_dict(calib), calib_json_path)
    print(f"[pipeline] calib JSON → {calib_json_path}")

    if calib.grade == CalibGrade.ABSTAIN:
        summary = {
            "image":        image_path,
            "grade":        "abstain",
            "n_kp":         calib.n_kp,
            "reproj_err_px": calib.reproj_err,
            "n_players":    None,
            "n_in_bounds":  None,
            "n_oob":        None,
            "elapsed_s":    round(time.time() - t0, 1),
            "calib_json":   calib_json_path,
        }
        print("[pipeline] ABSTAIN — calibration gate failed; no player positions emitted.")
        save_json(summary, os.path.join(output_dir, f"{stem}_summary.json"))
        return summary

    # ── 3. player detection ───────────────────────────────────────────────────
    print("[pipeline] detecting players …")
    players, team_labels, det_summary = player_detection.run(
        image_path, output_dir=output_dir, api_key=api_key,
    )
    n_players = len(players)
    print(f"[pipeline] {n_players} players detected "
          f"(team 0: {det_summary['team_counts']['team_0']}, "
          f"team 1: {det_summary['team_counts']['team_1']})")

    # ── 4. project to metric pitch coordinates ────────────────────────────────
    print("[pipeline] projecting foot pixels → world coordinates …")
    projected = project_players(players, team_labels.tolist(), calib.H)

    n_valid   = sum(1 for p in projected if p.world_xy is not None and not p.oob)
    n_oob     = sum(1 for p in projected if p.oob)
    n_fail    = sum(1 for p in projected if p.world_xy is None)
    print(f"[pipeline] in-bounds={n_valid}  oob={n_oob}  projection-failures={n_fail}")

    # serialise projections
    proj_records = []
    for p in projected:
        proj_records.append({
            "det_index":  p.det_index,
            "team":       p.team,
            "conf":       round(p.conf, 4),
            "pixel_u":    round(p.pixel_uv[0], 1),
            "pixel_v":    round(p.pixel_uv[1], 1),
            "world_Xw":   round(p.world_xy[0], 3) if p.world_xy else None,
            "world_Yw":   round(p.world_xy[1], 3) if p.world_xy else None,
            "oob":        p.oob,
        })
    save_json({"image": image_path, "players": proj_records}, proj_json_path)
    print(f"[pipeline] projections JSON → {proj_json_path}")

    # ── 5. top-down plot ──────────────────────────────────────────────────────
    print("[pipeline] generating top-down plot …")
    plot_topdown(
        projected,
        output_path=topdown_png_path,
        title=f"Player positions — {os.path.basename(image_path)}",
    )

    # ── 6. summary ────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    summary = {
        "image":         image_path,
        "grade":         calib.grade.value,
        "n_kp":          calib.n_kp,
        "reproj_err_px": round(calib.reproj_err, 3),
        "pan_deg":       calib.pan_deg,
        "tilt_deg":      calib.tilt_deg,
        "roll_deg":      calib.roll_deg,
        "n_players":     n_players,
        "n_in_bounds":   n_valid,
        "n_oob":         n_oob,
        "n_proj_fail":   n_fail,
        "elapsed_s":     elapsed,
        "calib_json":    calib_json_path,
        "proj_json":     proj_json_path,
        "topdown_png":   topdown_png_path,
    }
    save_json(summary, os.path.join(output_dir, f"{stem}_summary.json"))
    print(f"[pipeline] done in {elapsed}s")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="PnLCalib player-positioning pipeline (approach_4)."
    )
    ap.add_argument("image", help="Path to input broadcast frame")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for output artefacts (default: outputs/approach_4/<stem>/)")
    ap.add_argument("--api-key", default=None,
                    help="Roboflow API key (or ROBOFLOW_API_KEY env var)")
    ap.add_argument("--device", default=calib_mod.DEVICE,
                    help="PyTorch device (default: cuda:0)")
    args = ap.parse_args()

    key = (args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")) or None
    summary = run(args.image, args.output_dir, api_key=key, device=args.device)

    print("\n── PIPELINE SUMMARY ─────────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
