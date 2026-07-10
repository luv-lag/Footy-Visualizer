"""Stage 2: run pitchcal on the same frames, then score + visualise BOTH.

Run with the project venv:
    .venv/bin/python eval/eval_comparison/run_comparison.py

Fairness notes
--------------
* Both systems' homographies are converted to one common frame (pitch
  corner-origin, 105 x 68 m, pitch -> image pixels) before anything is
  computed.  PnLCalib's H is centre-origin world -> image (documented in
  pipeline/pnlcalib_wrapper/calibration.py), so it is right-composed with
  the translation T: (x, y) -> (x - 52.5, y - 34).
* Overlays and top-down warps for both systems are rendered by the SAME
  shared code in this script (not by either system's own renderer).
* The quantitative scores (trimmed chamfer "fit", lower better, and
  "explained" fraction, higher better) come from pitchcal's Scorer, applied
  identically to both homographies on identical detected-marking evidence.
  CAVEAT: this yardstick was built and tuned around pitchcal, so it is not
  perfectly neutral; it is used because it is the only shared quantitative
  yardstick available, and the visual checks are the primary evidence.
"""

from __future__ import annotations

import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))  # .../eval/eval_comparison
ROOT = os.path.dirname(os.path.dirname(HERE))        # project root
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from pitchcal import calibrate  # noqa: E402
from pitchcal.detect import detect_features  # noqa: E402
from pitchcal.geometry import apply_h_w  # noqa: E402
from pitchcal.pitch_model import PitchModel  # noqa: E402
from pitchcal.solve import Scorer  # noqa: E402

OUT = os.path.join(HERE, "out")

def discover_frames() -> list[tuple[str, str]]:
    """All top-level images in data/samples/, with short stable slugs."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "samples", "*.png"))
                    + glob.glob(os.path.join(ROOT, "data", "samples", "*.jpg"))):
        base = os.path.basename(p)
        m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})", base)
        if m:
            slug = f"scr_{m.group(2)}{m.group(3)}{m.group(4)}"
        else:
            slug = os.path.splitext(base)[0][:12].replace(" ", "_")
        out.append((slug, os.path.relpath(p, ROOT)))
    return out


# Corner-origin -> PnLCalib centre-origin change of pitch coordinates:
#     corner (x, y) -> (x - L/2, W/2 - y)
# NOTE the y-flip: PnLCalib's world +Y points the OPPOSITE way to the corner
# frame's +Y.  The naive translation (x - L/2, y - W/2) silently mirrors
# every position across the pitch's long axis -- invisible in overlays and
# top-downs (pitch markings are symmetric under that flip) but caught by the
# handedness check.  Derivation: eval_comparison/validate_orientation_check.py.
T_CENTER_TO_CORNER = np.array([[1.0, 0.0, -52.5], [0.0, -1.0, 34.0], [0.0, 0.0, 1.0]])


def draw_overlay(img: np.ndarray, H: np.ndarray, pitch: PitchModel, color) -> np.ndarray:
    """Project template curves through H (corner-frame pitch -> image px)."""
    vis = img.copy()
    h, w = vis.shape[:2]
    for _, pts in pitch.sample_all(step=0.35):
        proj, wz = apply_h_w(H, pts)
        p = np.round(proj).astype(int)
        for i in range(len(p) - 1):
            if wz[i] > 1e-9 and wz[i + 1] > 1e-9 and np.abs(p[i + 1] - p[i]).max() < 150:
                a = p[i]
                if -w < a[0] < 2 * w and -h < a[1] < 2 * h:
                    cv2.line(vis, tuple(p[i]), tuple(p[i + 1]), color, 2, cv2.LINE_AA)
    return vis


def topdown(img: np.ndarray, H: np.ndarray, pitch: PitchModel, px_per_m: float = 10.0) -> np.ndarray:
    """Warp the frame to a bird's-eye view using H (corner-frame -> image)."""
    out_w, out_h = int(pitch.length * px_per_m), int(pitch.width * px_per_m)
    gx, gy = np.meshgrid(np.arange(out_w) / px_per_m, np.arange(out_h) / px_per_m)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    proj, wz = apply_h_w(H, pts)
    proj[wz <= 1e-9] = -1e6  # behind camera: map outside the frame
    mx = proj[:, 0].reshape(out_h, out_w).astype(np.float32)
    my = proj[:, 1].reshape(out_h, out_w).astype(np.float32)
    warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(30, 30, 30))
    # Draw the template on top so scale/orientation is checkable.
    for _, tpts in pitch.sample_all(step=0.5):
        q = np.round(tpts * px_per_m).astype(int)
        for i in range(len(q) - 1):
            if np.abs(q[i + 1] - q[i]).max() < 40:
                cv2.line(warped, tuple(q[i]), tuple(q[i + 1]), (0, 140, 255), 1)
    return warped


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pitch = PitchModel()
    with open(os.path.join(OUT, "pnlcalib_results.json")) as f:
        pnl = json.load(f)

    report: dict = {}
    for name, rel in discover_frames():
        img = cv2.imread(os.path.join(ROOT, rel))
        rec: dict = {"frame": name}

        # ---- pitchcal (public API, fresh run) ---------------------------- #
        res = calibrate(img)
        rec["pitchcal"] = {
            "status": res.status.value,
            "confidence": round(res.confidence, 3),
        }
        H_pc: np.ndarray | None = None
        if res.ok:
            assert res.dm is not None and abs(res.dm.lam) < 1e-9, "lens model active"
            H_pc = res.H / res.H[2, 2]
            np.save(os.path.join(OUT, f"H_pitchcal_{name}.npy"), H_pc)
            rec["pitchcal"]["H"] = H_pc.tolist()
        else:
            rec["pitchcal"]["failure_reason"] = res.diagnostics.get(
                "reason", res.status.value
            )
            rec["pitchcal"]["diag_best"] = res.diagnostics.get("best")

        # ---- PnLCalib (from stage 1) ------------------------------------- #
        H_pn: np.ndarray | None = None
        prec = pnl[name]
        rec["pnlcalib"] = {
            "grade": prec["grade"],
            "n_kp": prec["n_kp"],
            "reproj_err": round(prec["reproj_err"], 2),
        }
        if prec["grade"] == "pass":
            H_world = np.load(os.path.join(OUT, f"H_pnlcalib_{name}.npy"))
            H_pn = H_world @ T_CENTER_TO_CORNER
            H_pn = H_pn / H_pn[2, 2]

        # ---- shared evidence + identical scoring ------------------------- #
        features = detect_features(img)
        scorer = Scorer(features, pitch)
        for tag, H in (("pitchcal", H_pc), ("pnlcalib", H_pn)):
            if H is None:
                continue
            fit, n_in = scorer.fit(H)
            expl = scorer.explained(H)
            rec[tag]["score_fit"] = round(float(fit), 4)  # lower = better
            rec[tag]["score_fit_px"] = round(float(fit) * scorer.cap / scorer.scale, 2)
            rec[tag]["score_explained"] = round(float(expl), 3)  # higher = better
            rec[tag]["n_template_samples_in_frame"] = int(n_in)
            color = (0, 220, 255) if tag == "pitchcal" else (255, 80, 255)
            ov = draw_overlay(img, H, pitch, color)
            cv2.putText(
                ov,
                f"{tag}  fit={rec[tag]['score_fit_px']:.1f}px expl={expl:.2f}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                3,
            )
            cv2.imwrite(os.path.join(OUT, f"{name}_{tag}_overlay.jpg"), ov)
            cv2.imwrite(
                os.path.join(OUT, f"{name}_{tag}_topdown.jpg"), topdown(img, H, pitch)
            )

        report[name] = rec
        print(json.dumps(rec, default=str)[:400], flush=True)

    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nsaved", os.path.join(OUT, "comparison.json"))


if __name__ == "__main__":
    main()
