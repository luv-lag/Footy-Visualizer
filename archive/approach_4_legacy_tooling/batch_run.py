"""
batch_run.py — run approach_4 pipeline on a list of images and print a results table.

Usage:
    python approach_4/batch_run.py --images <glob-or-list> [--device cuda:0]
    python approach_4/batch_run.py --synloc-random <N>   [--device cuda:0]

Run from football-analysis/ repo root with PnLCalib venv.
"""
from __future__ import annotations
import argparse
import glob
import os
import random
import sys
import time
import traceback

# repo root on path
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from approach_4 import calibration as calib_mod
from approach_4.calibration   import load_models, calibrate, CalibGrade
from approach_4.projection    import project_players
from approach_4.visualization import plot_topdown
from approach_4.io_utils      import save_json, calib_result_to_dict
from approach_4 import player_detection

import cv2

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_SYNLOC_DIR = os.path.join(_REPO_ROOT, "data", "SoccerNet", "SpiideoSynLoc", "val")


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in _IMAGE_EXTS


def run_one(image_path: str, model_kp, model_l, output_dir: str,
            topdown_dir: str, api_key: str | None, device: str) -> dict:
    """Run the full pipeline on one image; return a result row."""
    stem = _stem(image_path)
    out  = os.path.join(output_dir, stem)
    os.makedirs(out, exist_ok=True)

    frame = cv2.imread(image_path)
    if frame is None:
        return {"image": os.path.basename(image_path), "error": "unreadable"}

    t0    = time.time()
    calib = calibrate(frame, model_kp, model_l, device=device)

    save_json(calib_result_to_dict(calib),
              os.path.join(out, f"{stem}_calib.json"))

    row = {
        "image":         os.path.basename(image_path),
        "grade":         calib.grade.value,
        "n_kp":          calib.n_kp,
        "reproj_px":     round(calib.reproj_err, 2),
        "pan_deg":       round(calib.pan_deg,  1) if calib.pan_deg  is not None else None,
        "tilt_deg":      round(calib.tilt_deg, 1) if calib.tilt_deg is not None else None,
        "n_players":     None,
        "n_in_bounds":   None,
        "n_oob":         None,
        "topdown":       None,
        "elapsed_s":     None,
        "error":         None,
    }

    if calib.grade == CalibGrade.ABSTAIN:
        row["elapsed_s"] = round(time.time() - t0, 1)
        return row

    try:
        players, team_labels, _ = player_detection.run(
            image_path, output_dir=out, api_key=api_key
        )
    except Exception as e:
        row["error"]    = f"detection: {e}"
        row["elapsed_s"] = round(time.time() - t0, 1)
        return row

    projected = project_players(players, team_labels.tolist(), calib.H)

    n_valid = sum(1 for p in projected if p.world_xy is not None and not p.oob)
    n_oob   = sum(1 for p in projected if p.oob)

    topdown_path = os.path.join(topdown_dir, f"{stem}_topdown.png")
    try:
        plot_topdown(projected, topdown_path,
                     title=f"Player positions — {os.path.basename(image_path)}")
    except Exception as e:
        row["error"] = f"viz: {e}"

    row.update({
        "n_players":   len(players),
        "n_in_bounds": n_valid,
        "n_oob":       n_oob,
        "topdown":     topdown_path,
        "elapsed_s":   round(time.time() - t0, 1),
    })

    proj_records = [{
        "det_index": p.det_index, "team": p.team,
        "world_Xw": round(p.world_xy[0], 3) if p.world_xy else None,
        "world_Yw": round(p.world_xy[1], 3) if p.world_xy else None,
        "oob": p.oob,
    } for p in projected]
    save_json({"image": image_path, "players": proj_records},
              os.path.join(out, f"{stem}_projections.json"))

    return row


def print_table(rows: list[dict]) -> None:
    cols = ["image", "grade", "n_kp", "reproj_px", "pan_deg", "tilt_deg",
            "n_players", "n_in_bounds", "n_oob", "elapsed_s", "error"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    sep  = "  ".join("-" * widths[c] for c in cols)
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + head)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print()

    n_pass    = sum(1 for r in rows if r.get("grade") == "pass")
    n_abstain = sum(1 for r in rows if r.get("grade") == "abstain")
    n_err     = sum(1 for r in rows if r.get("error"))
    players_list  = [r["n_players"]   for r in rows if r.get("n_players") is not None]
    inbounds_list = [r["n_in_bounds"] for r in rows if r.get("n_in_bounds") is not None]

    import statistics
    print(f"PASS={n_pass}  ABSTAIN={n_abstain}  ERRORS={n_err}  total={len(rows)}")
    if players_list:
        print(f"Players detected: median={statistics.median(players_list):.0f}  "
              f"min={min(players_list)}  max={max(players_list)}")
    if inbounds_list:
        print(f"In-bounds:        median={statistics.median(inbounds_list):.0f}  "
              f"min={min(inbounds_list)}  max={max(inbounds_list)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="Image paths or globs")
    ap.add_argument("--samples-dir", default=None,
                    help="Run on all images in this directory")
    ap.add_argument("--synloc-random", type=int, default=0,
                    help="Pick N random images from SynLoc val set")
    ap.add_argument("--output-dir", default=None,
                    help="Root for output artefacts")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default=calib_mod.DEVICE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # collect images
    image_paths: list[str] = []

    if args.samples_dir:
        for f in sorted(os.listdir(args.samples_dir)):
            full = os.path.join(args.samples_dir, f)
            if _is_image(full):
                image_paths.append(full)

    if args.images:
        for pat in args.images:
            for p in sorted(glob.glob(pat)):
                if _is_image(p):
                    image_paths.append(os.path.abspath(p))

    if args.synloc_random:
        random.seed(args.seed)
        all_synloc = sorted(
            os.path.join(_SYNLOC_DIR, f)
            for f in os.listdir(_SYNLOC_DIR)
            if _is_image(os.path.join(_SYNLOC_DIR, f))
        )
        image_paths.extend(random.sample(all_synloc, min(args.synloc_random, len(all_synloc))))

    if not image_paths:
        ap.error("No images specified. Use --samples-dir, --images, or --synloc-random.")

    output_dir = args.output_dir or os.path.join(_REPO_ROOT, "outputs", "approach_4", "batch")
    os.makedirs(output_dir, exist_ok=True)
    topdown_dir = os.path.join(output_dir, "topdowns")
    os.makedirs(topdown_dir, exist_ok=True)
    api_key = (args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")) or None

    print(f"[batch] loading PnLCalib models (device={args.device}) …")
    model_kp, model_l = load_models(device=args.device)

    print(f"[batch] running on {len(image_paths)} images …\n")
    rows: list[dict] = []
    for i, img in enumerate(image_paths, 1):
        print(f"── [{i}/{len(image_paths)}] {os.path.basename(img)}")
        try:
            row = run_one(img, model_kp, model_l, output_dir, topdown_dir, api_key, args.device)
        except Exception:
            row = {"image": os.path.basename(img), "error": traceback.format_exc()[-120:]}
        rows.append(row)
        print(f"   grade={row.get('grade','?')}  n_kp={row.get('n_kp','?')}  "
              f"players={row.get('n_players','?')}  in-bounds={row.get('n_in_bounds','?')}")

    print_table(rows)
    save_json({"runs": rows}, os.path.join(output_dir, "batch_results.json"))
    print(f"[batch] results JSON → {output_dir}/batch_results.json")


if __name__ == "__main__":
    main()
