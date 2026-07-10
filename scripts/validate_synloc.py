"""Validation on the Spiideo SynLoc benchmark (real ground truth).

Uses the person annotations: each annotated person has an image keypoint at
their ground contact and a ground-truth pitch position.  For every frame
the pipeline calibrates from pixels alone; predicted person positions are
compared against ground truth in metres.

Orientation is checked honestly: the ground-truth camera side (from the
annotation extrinsics) determines the expected canonical frame, and a
result is counted as flipped when it matches the 180-degree-rotated ground
truth instead.  Errors are reported against the canonical ground truth,
not best-of-both.

Usage: .venv/bin/python scripts/validate_synloc.py [n_frames] [--seed S]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from pitchcal import calibrate  # noqa: E402

ROOT = Path("Samples/data/SoccerNet/SpiideoSynLoc")


def load_annotations() -> tuple[list[dict], dict[int, list[dict]]]:
    data = json.loads((ROOT / "annotations" / "val.json").read_text())
    by_img: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    return data["images"], by_img


def gt_camera_side_positive_y(im: dict) -> bool:
    """True if the GT camera sits on the +y side of the pitch centre."""
    P = np.array(im["camera_matrix"])  # 3x4 [R|t], world -> camera
    R, t = P[:, :3], P[:, 3]
    C = -R.T @ t
    return bool(C[1] > 0)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    seed = 11
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    images, by_img = load_annotations()
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(images), size=min(n, len(images)), replace=False)

    rows = []
    t_total = 0.0
    for k in picks:
        im = images[int(k)]
        anns = by_img.get(im["id"], [])
        path = ROOT / "val" / im["file_name"]
        frame = cv2.imread(str(path))
        if frame is None or len(anns) < 4:
            continue
        # Annotations are in the annotation resolution; images on disk may
        # be downscaled.
        sx = frame.shape[1] / im["width"]
        sy = frame.shape[0] / im["height"]
        feet_px = []
        feet_gt = []
        for a in anns:
            kp = a["keypoints"][-1]  # bottom (ground contact) point
            feet_px.append([kp[0] * sx, kp[1] * sy])
            feet_gt.append(a["position_on_pitch"])
        feet_px = np.array(feet_px)
        feet_gt = np.array(feet_gt)
        # Their frame: centre origin.  Ours: corner origin, 105 x 68.
        gt_corner = feet_gt + [52.5, 34.0]
        if gt_camera_side_positive_y(im):
            # camera on +y side -> canonical frame is the rot180 of theirs
            gt_canon = np.stack([105.0 - gt_corner[:, 0], 68.0 - gt_corner[:, 1]], axis=1)
        else:
            gt_canon = gt_corner

        t0 = time.time()
        res = calibrate(frame, player_feet=feet_px)
        dt = time.time() - t0
        t_total += dt
        row: dict = {"file": im["file_name"], "status": res.status.value, "t": round(dt, 2)}
        if res.ok:
            est = res.image_to_pitch(feet_px)
            # Only score persons plausibly on the pitch (GT within bounds).
            on = (
                (gt_canon[:, 0] > -2)
                & (gt_canon[:, 0] < 107)
                & (gt_canon[:, 1] > -2)
                & (gt_canon[:, 1] < 70)
            )
            if on.sum() < 4:
                on = np.ones(len(gt_canon), bool)
            err = np.linalg.norm(est - gt_canon, axis=1)[on]
            gt_flip = np.stack([105.0 - gt_canon[:, 0], 68.0 - gt_canon[:, 1]], axis=1)
            err_flip = np.linalg.norm(est - gt_flip, axis=1)[on]
            flipped = bool(np.median(err_flip) < np.median(err))
            row |= {
                "med_m": round(float(np.median(err)), 2),
                "p90_m": round(float(np.percentile(err, 90)), 2),
                "flipped": flipped,
                "conf": round(res.confidence, 2),
                "strategy": res.diagnostics["best"]["strategy"],
                "lambda": round(res.diagnostics["lambda"], 3),
                "players_used": res.used_player_formation,
            }
        rows.append(row)
        print(json.dumps(row))

    ok_rows = [r for r in rows if r["status"] == "ok"]
    good = [r for r in ok_rows if not r["flipped"]]
    print("\n================ SynLoc SUMMARY ================")
    print(f"frames: {len(rows)}   calibrated(OK): {len(ok_rows)} ({100*len(ok_rows)/max(len(rows),1):.0f}%)")
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("status breakdown:", statuses)
    if ok_rows:
        print(f"orientation correct: {len(good)}/{len(ok_rows)} "
              f"({100*len(good)/len(ok_rows):.0f}%)")
        meds = [r["med_m"] for r in good]
        if meds:
            print(f"player-position error (correct-orientation OK frames):")
            print(f"  median-of-medians: {np.median(meds):.2f} m   mean: {np.mean(meds):.2f} m")
            print(f"  p90 of per-frame medians: {np.percentile(meds, 90):.2f} m   worst: {np.max(meds):.2f} m")
        bad_ok = [r for r in ok_rows if r["flipped"] or r["med_m"] > 6.0]
        print(f"bad-but-reported-OK (flip or >6 m): {len(bad_ok)}")
    print(f"avg time/frame: {t_total/max(len(rows),1):.2f}s")


if __name__ == "__main__":
    main()
