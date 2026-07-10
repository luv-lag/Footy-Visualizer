"""Synthetic ground-truth validation of the pitchcal pipeline.

Renders a varied suite of broadcast-like views with exact ground-truth
homographies (both camera sides, several shot types, optional lens
distortion), runs the full pipeline, and reports:

* calibration rate and status breakdown,
* orientation correctness -- whether the 180-degree symmetry was resolved
  to the canonical (camera-side) frame, checked against the true camera
  position, per frame,
* position error in metres over the visible frame area (median / p90),
* honesty -- how often a *wrong* mapping was reported as OK vs flagged.

Usage:  .venv/bin/python eval/scripts/validate_synthetic.py [n_frames] [--distort]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../eval/scripts
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))  # project root
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "pipeline"))

from pitchcal import CalibrationStatus, calibrate  # noqa: E402
from pitchcal.synthetic import SyntheticFrame, broadcast_camera_suite  # noqa: E402


def canonical_gt_mapping(fr: SyntheticFrame, pts_px: np.ndarray) -> np.ndarray:
    """GT image->pitch mapping expressed in pitchcal's canonical frame."""
    gt = fr.gt_image_to_pitch(pts_px)
    if fr.camera_pos[1] > 34.0:  # camera on the far side: canonical flips
        gt = np.stack([105.0 - gt[:, 0], 68.0 - gt[:, 1]], axis=1)
    return gt


def frame_errors(fr: SyntheticFrame, res) -> tuple[float, float, bool]:
    """(median_err_m, p90_err_m, orientation_flipped) over a frame grid."""
    w, h = fr.image.shape[1], fr.image.shape[0]
    xs, ys = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 9), np.linspace(0.25 * h, 0.95 * h, 7))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    gt = canonical_gt_mapping(fr, pts)
    # Only evaluate where GT lands on / near the pitch (in-view pitch area).
    ok = (
        (gt[:, 0] > -5) & (gt[:, 0] < 110) & (gt[:, 1] > -5) & (gt[:, 1] < 73)
    )
    if ok.sum() < 8:
        ok = np.ones(len(gt), bool)
    est = res.image_to_pitch(pts)
    err = np.linalg.norm(est - gt, axis=1)[ok]
    gt_flip = np.stack([105.0 - gt[:, 0], 68.0 - gt[:, 1]], axis=1)
    err_flip = np.linalg.norm(est - gt_flip, axis=1)[ok]
    flipped = bool(np.median(err_flip) < np.median(err))
    return float(np.median(err)), float(np.percentile(err, 90)), flipped


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
    with_distortion = "--distort" in sys.argv
    frames = broadcast_camera_suite(n=n, with_distortion=with_distortion)

    rows = []
    t_total = 0.0
    for fr in frames:
        t0 = time.time()
        res = calibrate(fr.image)
        dt = time.time() - t0
        t_total += dt
        row: dict = {"label": fr.label, "status": res.status.value, "t": round(dt, 2)}
        if res.ok:
            med, p90, flipped = frame_errors(fr, res)
            row |= {
                "med_m": round(med, 2),
                "p90_m": round(p90, 2),
                "flipped": flipped,
                "conf": round(res.confidence, 2),
                "strategy": res.diagnostics["best"]["strategy"],
            }
        rows.append(row)
        print(json.dumps(row))

    ok_rows = [r for r in rows if r["status"] == "ok"]
    good = [r for r in ok_rows if not r["flipped"]]
    flip = [r for r in ok_rows if r["flipped"]]
    print("\n================ SUMMARY ================")
    print(f"frames: {len(rows)}   calibrated(OK): {len(ok_rows)} ({100*len(ok_rows)/len(rows):.0f}%)")
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("status breakdown:", statuses)
    if ok_rows:
        print(f"orientation correct: {len(good)}/{len(ok_rows)} "
              f"({100*len(good)/max(len(ok_rows),1):.0f}%)  flips: {len(flip)}")
        meds = [r["med_m"] for r in good]
        if meds:
            print(f"position error (correct-orientation OK frames): "
                  f"median {np.median(meds):.2f} m   mean {np.mean(meds):.2f} m   "
                  f"worst {np.max(meds):.2f} m")
        # Honesty: OK results that are actually bad (>4 m median or flipped).
        bad_ok = [r for r in ok_rows if r["flipped"] or r["med_m"] > 4.0]
        print(f"bad-but-reported-OK: {len(bad_ok)}  "
              f"({[r['label'] for r in bad_ok]})")
        confs_good = [r["conf"] for r in good if r["med_m"] <= 4.0]
        confs_bad = [r["conf"] for r in bad_ok]
        if confs_good:
            print(f"confidence  good: {np.mean(confs_good):.2f}   "
                  f"bad: {np.mean(confs_bad) if confs_bad else float('nan'):.2f}")
    print(f"avg time/frame: {t_total/len(rows):.2f}s")


if __name__ == "__main__":
    main()
