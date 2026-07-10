"""
diagnose_synloc_scale.py
Diagnostic on the 50-image synloc_calibration_eval.json run.
De-confounds scale mismatch from genuine calibration failures.

Steps:
  0 — Pin down SynLoc's true pitch model from source (not inference)
  1 — Fit effective scale; rebuild position error in matched frame
  2 — Re-classify worst frames with scale removed
  3 — Corrected position error vs pitch-centre distance (fisheye re-test)

Does NOT re-run the full 6,777-image eval. Re-calibrates only the 29
calibrated images from the 50-image JSON run to obtain H matrices.

Run with:
    vendor/PnLCalib/.venv/bin/python eval/synloc/diagnose_synloc_scale.py [--eval-json FILE]
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
from PIL import Image

# ── PnLCalib path setup ───────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))       # .../eval/synloc
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))     # project root
PNLCALIB_DIR = os.path.join(PROJECT_ROOT, "vendor", "PnLCalib")
sys.path.insert(0, PNLCALIB_DIR)
os.chdir(PNLCALIB_DIR)

import yaml
from model.cls_hrnet   import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

# ── Dataset / eval JSON paths ─────────────────────────────────────────────────
DATA_DIR          = os.path.join(PROJECT_ROOT, "data", "SoccerNet", "SpiideoSynLoc")
VAL_IMG_DIR       = os.path.join(DATA_DIR, "val")
VAL_ANN_FILE      = os.path.join(DATA_DIR, "annotations", "val.json")
DEFAULT_EVAL_JSON = os.path.join(PROJECT_ROOT, ".cache", "synloc_diagnostics", "synloc_calibration_eval.json")
DEFAULT_OUT_DIR   = os.path.join(PROJECT_ROOT, ".cache", "synloc_diagnostics")

# ── Source files for Step 0 evidence ─────────────────────────────────────────
# NOTE: sskit now lives in the consolidated vendor/PnLCalib/.venv (the
# standalone .venv-synloc was merged into it -- see README/venv notes).
SSKIT_CAMERA_SRC  = os.path.join(PNLCALIB_DIR,
    ".venv/lib/python3.11/site-packages/sskit/camera.py")
SYNLOC_REF_SRC    = os.path.join(PROJECT_ROOT,
    "archive/pitch_analysis/synloc_reference.py")

# ── PnLCalib inference settings ───────────────────────────────────────────────
DEVICE                 = "cuda:0"
KP_THRESHOLD           = 0.3434
LINE_THRESHOLD         = 0.7867
MIN_KP_FOR_VALID_CALIB = 4

# ── Pitch model constants (from source — see Step 0) ─────────────────────────
# SynLoc (authoritative, from sskit/camera.py project_on_ground defaults):
SL_WIDTH   = 70.0    # m  (short axis = SynLoc X)
SL_LENGTH  = 120.0   # m  (long  axis = SynLoc Y)
SL_HALF_X  = SL_WIDTH  / 2   # ±35 m
SL_HALF_Y  = SL_LENGTH / 2   # ±60 m

# PnLCalib world frame (standard FIFA 105 × 68 m):
PNL_HALF_X = 52.5   # ±52.5 m (length axis)
PNL_HALF_Y = 34.0   # ±34 m   (width axis)

# Theoretical scale factors derived from declared dimensions:
THEORY_S_LEN = PNL_HALF_X / SL_HALF_Y   # 52.5 / 60  = 0.8750  (length: SynLoc Y → PnLCalib X)
THEORY_S_WID = PNL_HALF_Y / SL_HALF_X   # 34.0 / 35  = 0.9714  (width:  SynLoc X → PnLCalib Y)

# Threshold for "well-calibrated" image (intrinsic error < this → used for scale fitting)
WELL_CAL_INTRINSIC_M = 2.0


# ════════════════════════════════════════════════════════════════════════════════
# Coordinate helpers (use variable scale for Step 1 and beyond)
# ════════════════════════════════════════════════════════════════════════════════

def pnlcalib_to_synloc(xp, yp, s_len, s_wid):
    """PnLCalib world (Xp, Yp) → SynLoc world (Xs, Ys).  Sign: both axes negated."""
    return -yp / s_wid, -xp / s_len


def project_i2w(H_inv, u, v):
    q = H_inv @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


# ════════════════════════════════════════════════════════════════════════════════
# PnLCalib model loading and calibration
# ════════════════════════════════════════════════════════════════════════════════

def load_models():
    cfg   = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))
    torch.cuda.init()
    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(torch.load("weights/SV_kp", map_location=DEVICE))
    model_kp.to(DEVICE).eval()
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load("weights/SV_lines", map_location=DEVICE))
    model_l.to(DEVICE).eval()
    return model_kp, model_l


def calibrate_image(frame_bgr, w_orig, h_orig, model_kp, model_l):
    """Returns (H_or_None, n_kp, reproj_err, mode)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = tvf.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    if img_t.size()[-1] != 960:
        img_t = T.Resize((540, 960))(img_t)
    img_t = img_t.to(DEVICE)
    _, _, h_net, w_net = img_t.size()
    cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
    with torch.no_grad():
        heatmaps   = model_kp(img_t)
        heatmaps_l = model_l(img_t)
    kp_coords   = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
    kp_dict     = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
    lines_dict  = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
    kp_dict, lines_dict = complete_keypoints(
        kp_dict[0], lines_dict[0], w=w_net, h=h_net, normalize=True)
    n_kp = len(kp_dict)
    cam.update(kp_dict, lines_dict)
    result = cam.heuristic_voting(refine_lines=False)
    if result is None:
        return None, n_kp, float("inf"), "fail"
    if n_kp < MIN_KP_FOR_VALID_CALIB:
        return None, n_kp, float(result["rep_err"]), "degenerate"
    cp = result["cam_params"]
    K  = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]],
                   [0, cp["y_focal_length"], cp["principal_point"][1]],
                   [0, 0, 1]], dtype=np.float64)
    R   = np.array(cp["rotation_matrix"], dtype=np.float64)
    pos = np.array(cp["position_meters"],  dtype=np.float64)
    t   = -(R @ pos)
    H   = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H  /= H[2, 2]
    return H, n_kp, float(result["rep_err"]), str(result["mode"])


# ════════════════════════════════════════════════════════════════════════════════
# STEP 0 — Authoritative SynLoc pitch model from source
# ════════════════════════════════════════════════════════════════════════════════

def step0_pitch_model(ann_data: dict) -> None:
    print()
    print("=" * 72)
    print("  STEP 0 — SynLoc pitch model: authoritative source evidence")
    print("=" * 72)

    print("\n  ── Evidence 1: sskit/camera.py  project_on_ground() ─────────────")
    print(f"  File: {SSKIT_CAMERA_SRC}")
    print("""  Signature (line 82):
    def project_on_ground(camera_matrix, dist_poly, image,
                          width=70, height=120, resolution=10, center=(0,0), ...):
        center = center - [width/2, height/2]   → [-35, -60]
        gnd = grid2d(width*res, height*res) / res + center
           ↳ gnd[:,0] = x ∈ [-35, +35]   (width axis, 70 m total)
           ↳ gnd[:,1] = y ∈ [-60, +60]   (length axis, 120 m total)
  These (x, y, z=0) triples feed directly into world_to_image(camera_matrix, ...)
  → position_on_pitch = [x, y] in exactly this frame (metres, origin=centre).""")

    print("\n  ── Evidence 2: pitch_analysis/synloc_reference.py ───────────────")
    print(f"  File: {SYNLOC_REF_SRC}")
    print("""  Line 99–104 (verbatim comment + code):
    # Pitch half-extents in the SynLoc meter frame. Pitch model = 70 (width) x 120
    # (length) m centred at origin; confirmed by GT ranges and sskit
    # project_on_ground defaults (width=70, height=120, center=(0,0)).
    PITCH_HALF_X = CONFIG.width  / 100.0 / 2.0   # 35.0  (short axis, SynLoc x)
    PITCH_HALF_Y = CONFIG.length / 100.0 / 2.0   # 60.0  (long  axis, SynLoc y)

  Also (line 35):
    "Only a fixed dihedral ambiguity (which goal is +y / mirror) separates the
     two conventions" — confirms origin=centre, Y=length axis.""")

    print("\n  ── Evidence 3: val.json annotation ranges (empirical cross-check) ─")
    anns = ann_data["annotations"]
    xs = [a["position_on_pitch"][0] for a in anns]
    ys = [a["position_on_pitch"][1] for a in anns]
    print(f"  X (width axis) : min={min(xs):+.3f}  max={max(xs):+.3f}  "
          f"(expected ±35 m, players can be slightly off-pitch)")
    print(f"  Y (length axis): min={min(ys):+.3f}  max={max(ys):+.3f}  "
          f"(expected ±60 m)")
    # Check 3D keypoints for z-scale
    kp3d_z = [a["keypoints_3d"][0][2] for a in anns[:500]]
    print(f"  keypoints_3d[0][2] (head z, first 500 anns): "
          f"range [{min(kp3d_z):.2f}, {max(kp3d_z):.2f}] m  "
          f"(confirms Z unit = metres)")

    print("\n  ── Verdict ──────────────────────────────────────────────────────")
    print(f"  SynLoc pitch model (AUTHORITATIVE):")
    print(f"    Dimensions  : {SL_WIDTH:.0f} m (width) × {SL_LENGTH:.0f} m (length)")
    print(f"    Origin      : pitch centre")
    print(f"    X axis      : width  (short axis), half-extent = ±{SL_HALF_X:.1f} m")
    print(f"    Y axis      : length (long  axis), half-extent = ±{SL_HALF_Y:.1f} m")
    print(f"    Units       : metres (confirmed by keypoints_3d Z values)")
    print(f"    Dihedral    : ±Y direction is ambiguous (which goal = +Y or −Y)")
    print()
    print(f"  Theoretical scale factors to PnLCalib (105×68 m FIFA model):")
    print(f"    SynLoc Y (length ±60m) → PnLCalib X (length ±52.5m):")
    print(f"      s_len = 52.5 / 60 = {THEORY_S_LEN:.6f}")
    print(f"    SynLoc X (width  ±35m) → PnLCalib Y (width  ±34m):")
    print(f"      s_wid = 34.0 / 35 = {THEORY_S_WID:.6f}")
    print(f"    Sign: BOTH axes negated (empirically verified; pitch orientations")
    print(f"          are antiparallel for this camera placement convention).")
    print()
    print(f"  NOTE: These THEORETICAL scale factors assume PnLCalib correctly maps")
    print(f"  its 105m FIFA model landmarks to the SynLoc 120m synthetic pitch.")
    print(f"  They may differ from EFFECTIVE scale factors (what PnLCalib actually")
    print(f"  produces), because the two pitches have different relative proportions")
    print(f"  for key features (penalty spot, arc, etc.). Step 1 fits the effective")
    print(f"  scale empirically from well-calibrated frames.")


# ════════════════════════════════════════════════════════════════════════════════
# Re-calibrate all 29 calibrated images → build per-image player data
# ════════════════════════════════════════════════════════════════════════════════

def recalibrate_all(eval_data, images_by_id, anns_by_img, model_kp, model_l):
    """
    Re-run PnLCalib on every calibrated image from the 50-image JSON run.
    Returns a list of dicts, one per image, with H and per-player raw data.
    """
    calibrated_records = [r for r in eval_data["per_image"]
                          if r["status"] == "calibrated"]
    print(f"\n  Re-calibrating {len(calibrated_records)} images (GPU)...")

    results = []
    for rec in calibrated_records:
        image_id = rec["image_id"]
        fname    = rec["file_name"]
        img_info = images_by_id[image_id]
        img_path = os.path.join(VAL_IMG_DIR, fname)
        w_orig   = int(img_info["width"])
        h_orig   = int(img_info["height"])

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"    {fname}: cannot read, skipping")
            continue

        H, n_kp, reproj_err, mode = calibrate_image(
            frame, w_orig, h_orig, model_kp, model_l)

        if H is None:
            print(f"    {fname}: re-calibration FAILED (n_kp={n_kp})")
            continue

        H_inv = np.linalg.inv(H)
        anns  = anns_by_img[image_id]

        players = []
        for a in anns:
            foot_kp = a["keypoints"][1]
            if int(foot_kp[2]) == 0:
                continue
            xs_gt = float(a["position_on_pitch"][0])
            ys_gt = float(a["position_on_pitch"][1])
            u_gt  = float(foot_kp[0])
            v_gt  = float(foot_kp[1])
            xp_est, yp_est = project_i2w(H_inv, u_gt, v_gt)
            players.append({
                "xs_gt": xs_gt, "ys_gt": ys_gt,
                "u_gt": u_gt,   "v_gt": v_gt,
                "xp_est": xp_est, "yp_est": yp_est,
            })

        results.append({
            "image_id":          image_id,
            "file_name":         fname,
            "intrinsic_median_m": rec.get("intrinsic_median_m"),
            "n_kp":              n_kp,
            "reproj_err":        reproj_err,
            "H":                 H,
            "players":           players,
        })
        print(f"    {fname}  kp={n_kp}  reproj={reproj_err:.2f}px  "
              f"players={len(players)}")

    print(f"  Re-calibrated: {len(results)}/{len(calibrated_records)} images OK")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Fit effective scale; recompute position errors
# ════════════════════════════════════════════════════════════════════════════════

def step1_scale(img_results: list[dict]) -> tuple[float, float]:
    print()
    print("=" * 72)
    print("  STEP 1 — Effective scale fitting and position error re-evaluation")
    print("=" * 72)

    # Separate well-calibrated images (intrinsic < WELL_CAL_INTRINSIC_M)
    well_cal = [r for r in img_results
                if r["intrinsic_median_m"] is not None
                and r["intrinsic_median_m"] < WELL_CAL_INTRINSIC_M]
    print(f"\n  Well-calibrated images (intrinsic < {WELL_CAL_INTRINSIC_M}m): "
          f"{len(well_cal)}/{len(img_results)}")

    # ── Fit effective scale factors ────────────────────────────────────────────
    # Model: xp_est = -ys_gt × s_len + eps_x
    #        yp_est = -xs_gt × s_wid + eps_y
    # No intercept — the mapping goes through the origin (both frames centred at pitch centre).
    # OLS without intercept: s_len = -Σ(xp_est × ys_gt) / Σ(ys_gt²)
    #                        s_wid = -Σ(yp_est × xs_gt) / Σ(xs_gt²)
    xp_all, ys_all = [], []
    yp_all, xs_all = [], []
    for r in well_cal:
        for p in r["players"]:
            if abs(p["ys_gt"]) > 1.0:   # skip near-centreline (ys≈0 is degenerate for s_len)
                xp_all.append(p["xp_est"])
                ys_all.append(p["ys_gt"])
            if abs(p["xs_gt"]) > 1.0:   # skip near-halfway (xs≈0 degenerate for s_wid)
                yp_all.append(p["yp_est"])
                xs_all.append(p["xs_gt"])

    xp_a = np.array(xp_all);  ys_a = np.array(ys_all)
    yp_a = np.array(yp_all);  xs_a = np.array(xs_all)

    s_len_fit = float(-np.dot(xp_a, ys_a) / np.dot(ys_a, ys_a))
    s_wid_fit = float(-np.dot(yp_a, xs_a) / np.dot(xs_a, xs_a))

    # Implied effective pitch dimensions from fitted scale
    sl_half_y_eff = PNL_HALF_X / s_len_fit   # effective SynLoc Y half-extent
    sl_half_x_eff = PNL_HALF_Y / s_wid_fit   # effective SynLoc X half-extent

    print(f"\n  ── Effective vs theoretical scale factors ─────────────────────")
    print(f"  {'Axis':<16}  {'Theoretical':>13}  {'Fitted':>10}  {'Δ':>8}")
    print(f"  {'──'*20}")
    print(f"  {'s_len (Y→X)':<16}  {THEORY_S_LEN:>13.6f}  {s_len_fit:>10.6f}  "
          f"{s_len_fit - THEORY_S_LEN:>+8.6f}")
    print(f"  {'s_wid (X→Y)':<16}  {THEORY_S_WID:>13.6f}  {s_wid_fit:>10.6f}  "
          f"{s_wid_fit - THEORY_S_WID:>+8.6f}")
    print()
    print(f"  Fitted scale implies effective PnLCalib-as-seen pitch:")
    print(f"    SynLoc Y half-extent: {SL_HALF_Y:.1f} m (declared)  →  "
          f"{sl_half_y_eff:.2f} m (effective PnLCalib sees)")
    print(f"    SynLoc X half-extent: {SL_HALF_X:.1f} m (declared)  →  "
          f"{sl_half_x_eff:.2f} m (effective PnLCalib sees)")

    # ── Recompute position errors for ALL images using fitted scale ──────────
    nom_errs, fit_errs = [], []
    for r in img_results:
        for p in r["players"]:
            xs_est_nom, ys_est_nom = pnlcalib_to_synloc(
                p["xp_est"], p["yp_est"], THEORY_S_LEN, THEORY_S_WID)
            xs_est_fit, ys_est_fit = pnlcalib_to_synloc(
                p["xp_est"], p["yp_est"], s_len_fit, s_wid_fit)
            err_nom = float(np.sqrt((xs_est_nom - p["xs_gt"])**2 +
                                    (ys_est_nom - p["ys_gt"])**2))
            err_fit = float(np.sqrt((xs_est_fit - p["xs_gt"])**2 +
                                    (ys_est_fit - p["ys_gt"])**2))
            # Cap extreme values from badly-calibrated images for aggregate stats
            p["err_nom"] = err_nom if err_nom < 300 else float("nan")
            p["err_fit"] = err_fit if err_fit < 300 else float("nan")
            if np.isfinite(p["err_nom"]): nom_errs.append(p["err_nom"])
            if np.isfinite(p["err_fit"]): fit_errs.append(p["err_fit"])

    nom_a = np.array(nom_errs)
    fit_a = np.array(fit_errs)

    print(f"\n  ── Position error: BEFORE vs AFTER scale correction ────────────")
    print(f"  {'Metric':<12}  {'Nominal (old)':>15}  {'Fitted (new)':>13}  "
          f"{'Improvement':>13}")
    print(f"  {'──'*20}")
    for label, pct, old_v, new_v in [
        ("Median",  50,  float(np.nanmedian(nom_a)), float(np.nanmedian(fit_a))),
        ("Mean",    None,float(np.nanmean(nom_a)),   float(np.nanmean(fit_a))),
        ("p90",     90,  float(np.nanpercentile(nom_a,90)), float(np.nanpercentile(fit_a,90))),
        ("p95",     95,  float(np.nanpercentile(nom_a,95)), float(np.nanpercentile(fit_a,95))),
    ]:
        delta = old_v - new_v
        print(f"  {label:<12}  {old_v:>13.2f} m  {new_v:>11.2f} m  {delta:>+11.2f} m")
    print(f"  n (nominal): {len(nom_a)}  n (fitted): {len(fit_a)}")

    pct_reduction = 100 * (np.nanmedian(nom_a) - np.nanmedian(fit_a)) / np.nanmedian(nom_a)
    print(f"\n  Median error reduction: {pct_reduction:.1f}% of original 15.8m")
    if pct_reduction > 50:
        print("  → Scale mismatch was the dominant error source.")
    elif pct_reduction > 20:
        print("  → Scale mismatch was a significant but not dominant contributor.")
    else:
        print("  → Scale mismatch explains little of the error; other sources dominate.")

    # ── Per-image breakdown ───────────────────────────────────────────────────
    print(f"\n  ── Per-image comparison (well-calibrated frames) ───────────────")
    print(f"  {'File':<18}  {'intr_m':>7}  {'pos_nom':>8}  {'pos_fit':>8}  {'Δ':>8}")
    print(f"  {'──'*20}")
    for r in sorted(img_results, key=lambda x: x.get("intrinsic_median_m") or 999):
        intr = r["intrinsic_median_m"]
        if intr is None or intr >= WELL_CAL_INTRINSIC_M:
            continue
        errs_nom = [p["err_nom"] for p in r["players"] if np.isfinite(p.get("err_nom", float("nan")))]
        errs_fit = [p["err_fit"] for p in r["players"] if np.isfinite(p.get("err_fit", float("nan")))]
        if not errs_nom or not errs_fit:
            continue
        old_m = float(np.median(errs_nom))
        new_m = float(np.median(errs_fit))
        print(f"  {r['file_name']:<18}  {intr:>7.2f}  {old_m:>8.2f}  {new_m:>8.2f}  "
              f"{old_m - new_m:>+8.2f}")

    return s_len_fit, s_wid_fit


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Re-classify worst frames with scale fixed
# ════════════════════════════════════════════════════════════════════════════════

TRANSFORMS = {
    "identity":    lambda xs, ys: (xs, ys),
    "negate_X":    lambda xs, ys: (-xs, ys),
    "negate_Y":    lambda xs, ys: (xs, -ys),
    "negate_both": lambda xs, ys: (-xs, -ys),
    "swap_XY":     lambda xs, ys: (ys, xs),
}

# Residual threshold for "fixed" vs "genuinely failed" after best transform
FIXED_THRESHOLD_M = 8.0
ORIGINAL_CATASTROPHIC_M = 20.0  # definition of "catastrophic" in original errors


def step2_reclassify(img_results: list[dict], s_len_fit: float, s_wid_fit: float,
                     n_worst: int = 8) -> None:
    print()
    print("=" * 72)
    print("  STEP 2 — Transform re-classification with scale corrected")
    print("=" * 72)
    print(f"  Scale: s_len={s_len_fit:.6f}  s_wid={s_wid_fit:.6f}  "
          f"(fitted from well-calibrated frames)")

    # Compute per-image median corrected position error
    for r in img_results:
        fit_errs = [p["err_fit"] for p in r["players"]
                    if np.isfinite(p.get("err_fit", float("nan")))]
        r["median_fit_err"] = float(np.median(fit_errs)) if fit_errs else float("nan")
        nom_errs = [p["err_nom"] for p in r["players"]
                    if np.isfinite(p.get("err_nom", float("nan")))]
        r["median_nom_err"] = float(np.median(nom_errs)) if nom_errs else float("nan")

    # Identify "catastrophic" in original errors (> 20m nominal)
    catastrophic = [r for r in img_results
                    if np.isfinite(r["median_nom_err"])
                    and r["median_nom_err"] > ORIGINAL_CATASTROPHIC_M]
    catastrophic.sort(key=lambda r: r["median_fit_err"])

    print(f"\n  Frames with original pos_err > {ORIGINAL_CATASTROPHIC_M}m: "
          f"{len(catastrophic)} images")
    print(f"\n  Testing transforms on all {len(catastrophic)} catastrophic frames "
          f"(with fitted scale):")

    verdicts = {}   # fname -> (best_transform, best_residual, classification)

    for r in catastrophic:
        fname = r["file_name"]
        intr  = r["intrinsic_median_m"]
        players = r["players"]
        if len(players) < 4:
            print(f"\n  {fname}: too few players ({len(players)}), skipping")
            continue

        # Corrected estimated positions (fitted scale)
        est_fit = np.array([pnlcalib_to_synloc(p["xp_est"], p["yp_est"],
                                                s_len_fit, s_wid_fit)
                            for p in players])
        gt = np.array([(p["xs_gt"], p["ys_gt"]) for p in players])

        print(f"\n  ── {fname}  nom={r['median_nom_err']:.1f}m  "
              f"fit={r['median_fit_err']:.1f}m  intr={intr or float('nan'):.1f}m  "
              f"kp={r['n_kp']} ──")

        transform_results = {}
        for tname, tfunc in TRANSFORMS.items():
            transformed = np.array([tfunc(xs, ys) for xs, ys in est_fit])
            errs = np.linalg.norm(transformed - gt, axis=1)
            transform_results[tname] = float(np.median(errs))

        best = min(transform_results, key=transform_results.get)
        best_residual = transform_results[best]

        for tname, merr in sorted(transform_results.items(), key=lambda x: x[1]):
            marker = " ← BEST" if tname == best else ""
            print(f"    {tname:<15}: {merr:7.2f} m{marker}")

        # Classify
        if best == "identity" and best_residual < FIXED_THRESHOLD_M:
            cls = "scale_artifact"
            desc = f"scale-only fix sufficient (residual {best_residual:.1f}m < {FIXED_THRESHOLD_M}m)"
        elif best != "identity" and best_residual < FIXED_THRESHOLD_M:
            cls = "disambiguation_flip"
            desc = (f"'{best}' transform fixes it after scale (residual {best_residual:.1f}m < "
                    f"{FIXED_THRESHOLD_M}m) → pitch-orientation flip")
        else:
            cls = "genuine_failure"
            desc = f"no transform helps (best residual {best_residual:.1f}m ≥ {FIXED_THRESHOLD_M}m)"

        verdicts[fname] = (best, best_residual, cls)
        print(f"    VERDICT: {cls.upper()} — {desc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = Counter(v[2] for v in verdicts.values())
    total  = len(verdicts)

    print()
    print("  ── Step 2 Summary ──────────────────────────────────────────────")
    print(f"  Of {len(catastrophic)} originally-catastrophic frames "
          f"({total} analysable):")
    print(f"    (a) scale_artifact      : {counts.get('scale_artifact', 0):>3}  "
          f"— fixed by scale alone; no convention issue")
    print(f"    (b) disambiguation_flip : {counts.get('disambiguation_flip', 0):>3}  "
          f"— fixed by axis flip + scale; expected on symmetric pitch views")
    print(f"    (c) genuine_failure     : {counts.get('genuine_failure', 0):>3}  "
          f"— no single transform helps; PnLCalib found a wrong pose")

    n_a = counts.get('scale_artifact', 0)
    n_b = counts.get('disambiguation_flip', 0)
    n_c = counts.get('genuine_failure', 0)

    print()
    if n_c == 0:
        print("  VERDICT: All catastrophic failures are explained by scale + orientation.")
        print("           No truly wrong calibrations — every bad frame is recoverable.")
    elif n_c == total:
        print("  VERDICT: All catastrophic failures are genuine wrong-pose calibrations.")
        print("           Scale correction does not help. PnLCalib found the wrong H.")
    else:
        print(f"  VERDICT: Mixed — {n_b} orientation flip(s), {n_c} genuinely wrong "
              f"calibration(s), {n_a} scale-only artifact(s).")
        if n_b > n_c:
            print("           Orientation disambiguation is the main tractable issue.")
        else:
            print("           Genuine calibration failures dominate; harder to fix.")

    # ── Mark for Step 3 ───────────────────────────────────────────────────────
    # Store verdict on each result for cross-referencing
    for r in img_results:
        v = verdicts.get(r["file_name"])
        r["verdict"] = v[2] if v else (
            "well_calibrated" if r.get("median_nom_err", 999) <= ORIGINAL_CATASTROPHIC_M
            else "unclassified")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Corrected position error vs pitch-centre distance
# ════════════════════════════════════════════════════════════════════════════════

def step3_pitch_radial(img_results: list[dict], out_dir: str) -> None:
    print()
    print("=" * 72)
    print("  STEP 3 — Corrected position error vs pitch-centre distance")
    print("=" * 72)

    all_pitch_dist = []
    all_fit_err    = []
    all_verdict    = []

    for r in img_results:
        verd = r.get("verdict", "well_calibrated")
        for p in r["players"]:
            if not np.isfinite(p.get("err_fit", float("nan"))):
                continue
            # GT distance from pitch centre in SynLoc world metres
            pitch_dist = float(np.sqrt(p["xs_gt"] ** 2 + p["ys_gt"] ** 2))
            all_pitch_dist.append(pitch_dist)
            all_fit_err.append(p["err_fit"])
            all_verdict.append(verd)

    pd_a  = np.array(all_pitch_dist)
    err_a = np.array(all_fit_err)
    verd_a = np.array(all_verdict)

    # Cap extreme outliers for stats
    p99   = float(np.percentile(err_a, 99))
    mask  = err_a <= p99

    corr_all = float(np.corrcoef(pd_a, err_a)[0, 1])
    corr_p99 = float(np.corrcoef(pd_a[mask], err_a[mask])[0, 1])

    # Correlation for well-calibrated subset only
    wc_mask = mask & (verd_a == "well_calibrated")
    if wc_mask.sum() > 10:
        corr_wc = float(np.corrcoef(pd_a[wc_mask], err_a[wc_mask])[0, 1])
        wc_note = f"  Correlation (well-calibrated only, n={wc_mask.sum()}): r={corr_wc:+.4f}"
    else:
        corr_wc = float("nan")
        wc_note = "  (too few well-calibrated points for separate correlation)"

    print(f"\n  Points total: {len(pd_a)}")
    print(f"  Pitch distance range: {pd_a.min():.1f} – {pd_a.max():.1f} m")
    print(f"  Corrected error range: {err_a.min():.2f} – {err_a.max():.2f} m")
    print(f"\n  Pearson r (all):        {corr_all:+.4f}")
    print(f"  Pearson r (≤p99 clip):  {corr_p99:+.4f}  [n={mask.sum()}]")
    print(wc_note)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    COLORS = {
        "well_calibrated":   "#4fc3f7",
        "scale_artifact":    "#a5d6a7",
        "disambiguation_flip":"#fff176",
        "genuine_failure":   "#ef9a9a",
        "unclassified":      "#bdbdbd",
    }
    LABELS = {
        "well_calibrated":   "well-calibrated",
        "scale_artifact":    "scale artifact",
        "disambiguation_flip":"orientation flip",
        "genuine_failure":   "genuine failure",
        "unclassified":      "unclassified",
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    for vname in COLORS:
        vm = mask & (verd_a == vname)
        if vm.sum() == 0:
            continue
        ax.scatter(pd_a[vm], err_a[vm], alpha=0.35, s=8,
                   c=COLORS[vname], label=LABELS[vname], rasterized=True)

    # Binned median
    n_bins = 14
    edges  = np.linspace(pd_a[mask].min(), pd_a[mask].max(), n_bins + 1)
    bx, by = [], []
    for i in range(n_bins):
        bm = mask & (pd_a >= edges[i]) & (pd_a < edges[i + 1])
        if bm.sum() >= 5:
            bx.append((edges[i] + edges[i + 1]) / 2)
            by.append(float(np.median(err_a[bm])))
    if bx:
        ax.plot(bx, by, "w-", lw=2.5, zorder=8, label="bin median")
        ax.plot(bx, by, "wo", ms=5, zorder=9)

    ax.set_xlabel("Player distance from pitch centre (metres, SynLoc GT)",
                  fontsize=11, color="white")
    ax.set_ylabel("Position error — corrected scale (metres)", fontsize=11, color="white")
    ax.set_title(
        f"Pitch-centre distance vs corrected position error\n"
        f"Pearson r = {corr_p99:+.3f}  (p99-clipped, n={mask.sum()})",
        color="white", fontsize=12)
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8, markerscale=2)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")

    plot_path = os.path.join(out_dir, "diag_pitch_radial.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Scatter plot saved → {plot_path}")

    print()
    r = corr_p99
    if r >= 0.35:
        print(f"  VERDICT: POSITIVE correlation (r={r:+.3f})")
        print("           Residual error grows with pitch-centre distance — consistent")
        print("           with pitch-model dimension mismatch at the ends (scale not")
        print("           perfectly corrected, or pitch markings non-uniform).")
    elif r <= -0.20:
        print(f"  VERDICT: NEGATIVE correlation (r={r:+.3f})")
        print("           Errors are worse near pitch centre. Unusual; may indicate")
        print("           the scale correction over-compensated for centre-field players.")
    else:
        print(f"  VERDICT: NO significant correlation (r={r:+.3f})")
        print("           Corrected position error is uniform across the pitch.")
        print("           Scale correction has removed the main systematic bias.")
        print("           Remaining error is random per-frame calibration noise.")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Scale diagnostic on 50-image SynLoc calibration run.")
    ap.add_argument("--eval-json", default=DEFAULT_EVAL_JSON)
    ap.add_argument("--ann-file",  default=VAL_ANN_FILE)
    ap.add_argument("--val-dir",   default=VAL_IMG_DIR)
    ap.add_argument("--out-dir",   default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    # Resolve all paths before os.chdir()
    args.eval_json = os.path.abspath(args.eval_json)
    args.ann_file  = os.path.abspath(args.ann_file)
    args.val_dir   = os.path.abspath(args.val_dir)
    args.out_dir   = os.path.abspath(args.out_dir)

    print(f"Eval JSON  : {args.eval_json}")
    with open(args.eval_json) as f:
        eval_data = json.load(f)

    print(f"Annotations: {args.ann_file}")
    with open(args.ann_file) as f:
        ann_data = json.load(f)

    images_by_id = {img["id"]: img for img in ann_data["images"]}
    anns_by_img  = defaultdict(list)
    for a in ann_data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    os.makedirs(args.out_dir, exist_ok=True)

    # STEP 0 — pitch model from source (no GPU needed)
    step0_pitch_model(ann_data)

    # Load GPU models (needed for Steps 1+2)
    print("\nLoading PnLCalib models...")
    model_kp, model_l = load_models()
    print("Models loaded.")

    # Re-calibrate all 29 calibrated images
    img_results = recalibrate_all(
        eval_data, images_by_id, anns_by_img, model_kp, model_l)

    # STEP 1 — fit scale and recompute errors
    s_len_fit, s_wid_fit = step1_scale(img_results)

    # STEP 2 — reclassify worst frames
    step2_reclassify(img_results, s_len_fit, s_wid_fit)

    # STEP 3 — pitch-centre distance scatter
    step3_pitch_radial(img_results, args.out_dir)

    print()
    print("=" * 72)
    print("  Diagnostic complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
