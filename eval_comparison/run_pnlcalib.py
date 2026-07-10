"""Stage 1: run PnLCalib (production approach_4 wrapper) on the 5 frames.

Run with the PnLCalib venv:
    ai-workspace/PnLCalib/.venv/bin/python eval_comparison/run_pnlcalib.py

Uses ONLY the documented approach_4 API (load_models / calibrate); no
modification of either system.  Saves, per frame, the full CalibResult
metadata as JSON and the homography as .npy (world centre-origin metres ->
image pixels, exactly as documented in approach_4/calibration.py).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ai-workspace", "football-analysis"))

from approach_4.calibration import CalibGrade, calibrate, load_models  # noqa: E402


OUT = os.path.join(HERE, "out")

def discover_frames() -> list[tuple[str, str]]:
    """All top-level images in Samples/, with short stable slugs."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "Samples", "*.png"))
                    + glob.glob(os.path.join(ROOT, "Samples", "*.jpg"))):
        base = os.path.basename(p)
        m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})", base)
        if m:
            slug = f"scr_{m.group(2)}{m.group(3)}{m.group(4)}"
        else:
            slug = os.path.splitext(base)[0][:12].replace(" ", "_")
        out.append((slug, os.path.relpath(p, ROOT)))
    return out



def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    device = "cuda:0"
    print("loading PnLCalib models ...", flush=True)
    model_kp, model_l = load_models(device)
    results = {}
    for name, rel in discover_frames():
        img = cv2.imread(os.path.join(ROOT, rel))
        assert img is not None, rel
        t0 = time.time()
        res = calibrate(img, model_kp, model_l, device=device)
        dt = time.time() - t0
        rec = {
            "frame": name,
            "path": rel,
            "grade": res.grade.value,
            "n_kp": res.n_kp,
            "reproj_err": res.reproj_err,
            "mode": res.mode,
            "time_s": round(dt, 2),
        }
        if res.grade == CalibGrade.PASS:
            np.save(os.path.join(OUT, f"H_pnlcalib_{name}.npy"), res.H)
            rec["H"] = res.H.tolist()
            rec["cam_pos"] = res.cam_pos.tolist()
            rec["pan_deg"] = res.pan_deg
            rec["tilt_deg"] = res.tilt_deg
        results[name] = rec
        print(json.dumps(rec)[:300], flush=True)
    with open(os.path.join(OUT, "pnlcalib_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved", os.path.join(OUT, "pnlcalib_results.json"))


if __name__ == "__main__":
    main()
