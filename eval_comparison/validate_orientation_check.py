"""Part-3 validation of the standalone orientation checker.

Pass 1 (synthetic, generality): take known-correct ground-truth homographies
from pitchcal's synthetic renderer, build their deliberately-wrong
orientation twins (rot180 / mirror-x / mirror-y), and feed ONLY the twins to
``validate_and_canonicalize_homography`` -- exactly as an external system's
output would arrive, with no pitchcal pipeline state.  The rot180 twin must
be corrected back to the canonical ground truth; the mirror twins must be
rejected as physically impossible.

Pass 2 (real impact): run every saved PnLCalib homography from the 11-frame
broadcast evaluation (eval_comparison/out/H_pnlcalib_*.npy, centre-origin
convention per approach_4/calibration.py) through the same function, report
per-frame outcomes, and render before/after top-down warps for any frame
where the 180-degree correction fired.

Run:  .venv/bin/python eval_comparison/validate_orientation_check.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from run_comparison import T_CENTER_TO_CORNER, topdown  # noqa: E402  (shared renderer)

from pitchcal import validate_and_canonicalize_homography  # noqa: E402
from pitchcal.geometry import apply_h_w, canonicalize  # noqa: E402
from pitchcal.pitch_model import PitchModel  # noqa: E402
from pitchcal.synthetic import broadcast_camera_suite  # noqa: E402

OUT = os.path.join(HERE, "out")


def mapping_diff_px(H1: np.ndarray, H2: np.ndarray, pitch: PitchModel) -> float:
    """Max pixel discrepancy of two pitch->image maps over a pitch grid."""
    gx, gy = np.meshgrid(np.linspace(5, pitch.length - 5, 9), np.linspace(5, pitch.width - 5, 7))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    p1, w1 = apply_h_w(H1, pts)
    p2, w2 = apply_h_w(H2, pts)
    ok = (np.abs(w1) > 1e-9) & (np.abs(w2) > 1e-9)
    return float(np.abs(p1[ok] - p2[ok]).max())


def synthetic_pass() -> bool:
    """Wrong-orientation twins of external-style GT homographies."""
    print("=== Pass 1: synthetic generality check ===")
    pitch = PitchModel()
    frames = broadcast_camera_suite(n=10)  # varied poses, both camera sides
    n_ok = n_total = 0
    for fr in frames:
        H_gt = fr.H_gt
        # The canonical form of the ground truth (what a correct answer is).
        H_ref, _ = canonicalize(H_gt / np.linalg.norm(H_gt), pitch.length, pitch.width)

        # --- rot180 twin: must be corrected back to H_ref ---------------- #
        twin = H_gt @ pitch.rot180()
        chk = validate_and_canonicalize_homography(twin, fr.image, pitch=pitch)
        d = mapping_diff_px(chk.corrected_H, H_ref, pitch) if chk.is_valid else float("inf")
        ok_rot = chk.is_valid and d < 1e-5
        n_total += 1
        n_ok += ok_rot
        cam_side = "south(y<0)" if fr.camera_pos[1] < 34 else "north(y>W)"
        print(
            f"  {fr.label:22} cam={cam_side}  rot180-twin: valid={chk.is_valid} "
            f"flipped={chk.was_flipped} diff_to_truth={d:.2e}px  "
            f"{'OK' if ok_rot else 'FAIL'}"
        )

        # --- mirror twins: must be REJECTED ------------------------------ #
        for mname, M in (("mirror-x", pitch.mirror_x()), ("mirror-y", pitch.mirror_y())):
            chk_m = validate_and_canonicalize_homography(H_gt @ M, fr.image, pitch=pitch)
            n_total += 1
            n_ok += (not chk_m.is_valid)
            if chk_m.is_valid:
                print(f"    FAIL: {mname} twin was accepted ({chk_m.reason})")
    print(f"  -> {n_ok}/{n_total} twin checks behaved correctly\n")
    return n_ok == n_total


# PnLCalib's ground-plane parity: feeding its homographies through the
# naive centre->corner translation (same +Y direction as pitchcal's corner
# frame) makes ALL of them fail the handedness test -- i.e. PnLCalib's
# world +Y points the OPPOSITE way.  The template-overlay evaluation could
# not see this (the marking layout is mirror-symmetric), but player
# positions mapped under the naive conversion would be mirrored across the
# pitch's long axis.  The parity-corrected conversion is:
#     corner (x, y)  ->  PnLCalib world (x - L/2, W/2 - y)
def m_parity_corrected(pitch: PitchModel) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, -pitch.length / 2.0], [0.0, -1.0, pitch.width / 2.0], [0.0, 0.0, 1.0]]
    )


def real_pass() -> None:
    """Every saved PnLCalib homography from the 11-frame evaluation."""
    print("=== Pass 2: PnLCalib homographies from the 11-frame evaluation ===")
    pitch = PitchModel()
    with open(os.path.join(OUT, "pnlcalib_results.json")) as f:
        pnl = json.load(f)

    # 2a. Naive conversion (documented '+-34 width' reading, +Y assumed to
    # match the corner frame): expected to expose the parity mismatch.
    n_rej = 0
    for name, rec in pnl.items():
        if rec["grade"] != "pass":
            continue
        H_world = np.load(os.path.join(OUT, f"H_pnlcalib_{name}.npy"))
        img = cv2.imread(os.path.join(ROOT, rec["path"]))
        chk = validate_and_canonicalize_homography(H_world, img, pitch=pitch, origin="center")
        n_rej += int(not chk.is_valid)
    print(
        f"  2a. naive centre-origin conversion: {n_rej}/11 rejected as "
        "mirror-image\n      -> PnLCalib's world +Y is opposite to the corner "
        "frame's +Y (a real convention hazard the overlays cannot show)."
    )

    # 2b. Parity-corrected conversion: the actual orientation audit.
    print("  2b. parity-corrected conversion (corner (x,y) -> (x-L/2, W/2-y)):")
    M = m_parity_corrected(pitch)
    counts = {"already_correct": 0, "corrected": 0, "rejected": 0}
    for name, rec in pnl.items():
        if rec["grade"] != "pass":
            print(f"    {name:14} (no homography saved -- PnLCalib abstained)")
            continue
        H_world = np.load(os.path.join(OUT, f"H_pnlcalib_{name}.npy"))
        img = cv2.imread(os.path.join(ROOT, rec["path"]))
        H_corner = H_world @ M
        chk = validate_and_canonicalize_homography(H_corner, img, pitch=pitch)
        if not chk.is_valid:
            outcome = "REJECTED"
            counts["rejected"] += 1
        elif chk.was_flipped:
            outcome = "CORRECTED (rot180)"
            counts["corrected"] += 1
        else:
            outcome = "already correct"
            counts["already_correct"] += 1
        print(f"    {name:14} {outcome:20} | {chk.reason[:60]}")
        if chk.is_valid and chk.was_flipped:
            # Before/after top-down warps for visual confirmation.
            H_before = H_corner / np.linalg.norm(H_corner)
            cv2.imwrite(
                os.path.join(OUT, f"{name}_orientfix_before_topdown.jpg"),
                topdown(img, H_before, pitch),
            )
            cv2.imwrite(
                os.path.join(OUT, f"{name}_orientfix_after_topdown.jpg"),
                topdown(img, chk.corrected_H, pitch),
            )
            print(f"        before/after top-downs saved: {name}_orientfix_*_topdown.jpg")
    print(f"\n  summary (parity-corrected): {counts}")


if __name__ == "__main__":
    ok = synthetic_pass()
    real_pass()
    if not ok:
        sys.exit(1)
