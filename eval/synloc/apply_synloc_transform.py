"""
apply_synloc_transform.py
Applies the authoritative SynLoc→PnLCalib transform (no fitting, no per-frame search).

Steps:
  1 — Resolve the sign convention globally once from well-calibrated frames
  2 — Apply fixed transform to ALL calibrated frames; aggregate error comparison
  3 — Good-calibration frames only: true position accuracy + flag dihedral flips
  4 — Three-way breakdown of the 50-image run

Run with:
    vendor/PnLCalib/.venv/bin/python eval/synloc/apply_synloc_transform.py [--eval-json FILE]
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict

import numpy as np
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

# ── PnLCalib inference settings ───────────────────────────────────────────────
DEVICE                 = "cuda:0"
KP_THRESHOLD           = 0.3434
LINE_THRESHOLD         = 0.7867
MIN_KP_FOR_VALID_CALIB = 4

# ── Authoritative pitch constants (Step 0 of previous pass) ──────────────────
# SynLoc:   X = width  ±35 m,  Y = length ±60 m,  origin = pitch centre, metres
# PnLCalib: X = length ±52.5 m, Y = width  ±34 m,  origin = pitch centre, metres
# Transform SynLoc → PnLCalib: axis SWAP + per-axis scale + sign to be resolved:
#   Xp = sign_len * s_len * Ys_gt    (SynLoc length Y → PnLCalib length X)
#   Yp = sign_wid * s_wid * Xs_gt    (SynLoc width  X → PnLCalib width  Y)
# Inverse (what we compute here, given Xp_est = H_inv @ pixel):
#   Ys_est = sign_len * Xp_est / s_len
#   Xs_est = sign_wid * Yp_est / s_wid
S_LEN = 52.5 / 60.0    # 0.875000  (length scale factor)
S_WID = 34.0  / 35.0   # 0.971429  (width  scale factor)

# Thresholds
WELL_CAL_INTRINSIC_M = 2.0    # for sign resolution (tight)
GOOD_CAL_INTRINSIC_M = 5.0    # for Step 3/4 split (broader: includes borderline frames)
FLIP_POSITION_M      = 5.0    # position error threshold to classify as "flipped pose"

# 4 sign combinations (sign_len, sign_wid) for the back-projection
SIGN_COMBOS = {
    "++": (+1, +1),
    "+-": (+1, -1),
    "-+": (-1, +1),
    "--": (-1, -1),   # our original convention from prior evaluation
}


# ════════════════════════════════════════════════════════════════════════════════
# PnLCalib model + calibration
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
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = tvf.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    if img_t.size()[-1] != 960:
        img_t = T.Resize((540, 960))(img_t)
    img_t = img_t.to(DEVICE)
    _, _, h_net, w_net = img_t.size()
    cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
    with torch.no_grad():
        hm  = model_kp(img_t)
        hml = model_l(img_t)
    kp_c  = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :])
    ln_c  = get_keypoints_from_heatmap_batch_maxpool_l(hml[:, :-1, :, :])
    kpd   = coords_to_dict(kp_c, threshold=KP_THRESHOLD)
    lnd   = coords_to_dict(ln_c, threshold=LINE_THRESHOLD)
    kpd, lnd = complete_keypoints(kpd[0], lnd[0], w=w_net, h=h_net, normalize=True)
    n_kp = len(kpd)
    cam.update(kpd, lnd)
    result = cam.heuristic_voting(refine_lines=False)
    if result is None:
        return None, n_kp, float("inf"), "fail"
    if n_kp < MIN_KP_FOR_VALID_CALIB:
        return None, n_kp, float(result["rep_err"]), "degenerate"
    cp = result["cam_params"]
    K = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]],
                  [0, cp["y_focal_length"], cp["principal_point"][1]],
                  [0, 0, 1]], dtype=np.float64)
    R   = np.array(cp["rotation_matrix"], dtype=np.float64)
    pos = np.array(cp["position_meters"],  dtype=np.float64)
    t   = -(R @ pos)
    H   = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H  /= H[2, 2]
    return H, n_kp, float(result["rep_err"]), str(result["mode"])


def project_i2w(H_inv, u, v):
    q = H_inv @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


# ════════════════════════════════════════════════════════════════════════════════
# Position error with a given sign combo
# ════════════════════════════════════════════════════════════════════════════════

def pos_err(xp_est, yp_est, xs_gt, ys_gt, sign_len, sign_wid):
    """Euclidean position error in SynLoc metres for one player."""
    xs_est = sign_wid * yp_est / S_WID
    ys_est = sign_len * xp_est / S_LEN
    return float(np.sqrt((xs_est - xs_gt) ** 2 + (ys_est - ys_gt) ** 2))


# ════════════════════════════════════════════════════════════════════════════════
# Re-calibrate all 29 calibrated images
# ════════════════════════════════════════════════════════════════════════════════

def recalibrate_all(eval_data, images_by_id, anns_by_img, model_kp, model_l):
    calibrated = [r for r in eval_data["per_image"] if r["status"] == "calibrated"]
    print(f"  Re-calibrating {len(calibrated)} images (GPU)...")
    results = []
    for rec in calibrated:
        image_id = rec["image_id"]
        fname    = rec["file_name"]
        img_info = images_by_id[image_id]
        img_path = os.path.join(VAL_IMG_DIR, fname)
        w_orig   = int(img_info["width"])
        h_orig   = int(img_info["height"])
        frame    = cv2.imread(img_path)
        if frame is None:
            print(f"    {fname}: cannot read")
            continue
        H, n_kp, reproj_err, mode = calibrate_image(
            frame, w_orig, h_orig, model_kp, model_l)
        if H is None:
            print(f"    {fname}: re-calibration failed (n_kp={n_kp})")
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
            players.append({"xs_gt": xs_gt, "ys_gt": ys_gt,
                            "xp_est": xp_est, "yp_est": yp_est})
        results.append({
            "image_id":          image_id,
            "file_name":         fname,
            "intrinsic_median_m": rec.get("intrinsic_median_m"),
            "n_kp":              n_kp,
            "reproj_err":        reproj_err,
            "players":           players,
        })
    print(f"  Re-calibrated: {len(results)}/{len(calibrated)} OK")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Resolve sign convention globally
# ════════════════════════════════════════════════════════════════════════════════

def step1_resolve_signs(img_results):
    print()
    print("=" * 72)
    print("  STEP 1 — Global sign convention resolution")
    print("=" * 72)
    print(f"\n  Authoritative transform (SynLoc → PnLCalib):")
    print(f"    Axis swap: SynLoc Y (length) → PnLCalib X; SynLoc X (width) → PnLCalib Y")
    print(f"    Scale (FIXED, from declared dimensions, no fitting):")
    print(f"      s_len = 52.5 / 60  = {S_LEN:.6f}  (length axis)")
    print(f"      s_wid = 34.0 / 35  = {S_WID:.6f}  (width  axis)")
    print(f"    Sign: 4 combinations tested, best chosen globally")
    print(f"\n  Sign resolution uses: frames with intrinsic error < {WELL_CAL_INTRINSIC_M} m")

    # Collect all players from well-calibrated frames
    well_cal = [r for r in img_results
                if r["intrinsic_median_m"] is not None
                and r["intrinsic_median_m"] < WELL_CAL_INTRINSIC_M]
    print(f"  Well-calibrated frames: {len(well_cal)}")

    # Per-combo: pool ALL per-player errors across all well-cal frames, take median
    combo_errors = {}
    for combo, (sl, sw) in SIGN_COMBOS.items():
        all_errs = []
        for r in well_cal:
            for p in r["players"]:
                e = pos_err(p["xp_est"], p["yp_est"],
                            p["xs_gt"], p["ys_gt"], sl, sw)
                all_errs.append(e)
        arr = np.array(all_errs)
        arr_finite = arr[np.isfinite(arr)]
        combo_errors[combo] = {
            "median": float(np.median(arr_finite)),
            "mean":   float(np.mean(arr_finite)),
            "n":      int(len(arr_finite)),
        }

    # Print audit table
    print(f"\n  Sign-combo audit (pooled over {combo_errors[list(combo_errors)[0]]['n']} players):")
    print(f"  {'Combo':<8}  {'sign_len':>9}  {'sign_wid':>9}  {'Median err':>12}  {'Mean err':>10}")
    print(f"  {'──'*20}")

    best_combo = min(combo_errors, key=lambda k: combo_errors[k]["median"])

    for combo in ["++", "+-", "-+", "--"]:
        sl, sw = SIGN_COMBOS[combo]
        ce = combo_errors[combo]
        marker = "  ← CHOSEN" if combo == best_combo else ""
        print(f"  {combo:<8}  {'+1' if sl>0 else '-1':>9}  {'+1' if sw>0 else '-1':>9}  "
              f"{ce['median']:>10.3f} m  {ce['mean']:>8.3f} m{marker}")

    chosen_sl, chosen_sw = SIGN_COMBOS[best_combo]
    print(f"\n  CHOSEN: '{best_combo}'  (sign_len={'+1' if chosen_sl>0 else '-1'},  "
          f"sign_wid={'+1' if chosen_sw>0 else '-1'})")
    print(f"  Back-projection formula:")
    print(f"    Ys_est = {'+' if chosen_sl>0 else '-'}Xp_est / {S_LEN}  "
          f"  (SynLoc length, metres)")
    print(f"    Xs_est = {'+' if chosen_sw>0 else '-'}Yp_est / {S_WID}  "
          f"  (SynLoc width,  metres)")

    return best_combo, chosen_sl, chosen_sw


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Apply fixed transform, aggregate comparison
# ════════════════════════════════════════════════════════════════════════════════

def step2_aggregate(img_results, sign_len, sign_wid):
    print()
    print("=" * 72)
    print("  STEP 2 — Aggregate position error: authoritative transform")
    print("=" * 72)

    all_errs = []
    for r in img_results:
        frame_errs = []
        for p in r["players"]:
            e = pos_err(p["xp_est"], p["yp_est"],
                        p["xs_gt"], p["ys_gt"], sign_len, sign_wid)
            if e < 300:   # cap extreme outliers from bad calibrations
                all_errs.append(e)
                frame_errs.append(e)
        r["auth_errs"]       = frame_errs
        r["auth_median_err"] = float(np.median(frame_errs)) if frame_errs else float("nan")

    arr = np.array(all_errs)

    # Reference values from previous passes (from script output)
    PREV_NOMINAL = 15.85   # original eval, nominal scale
    PREV_FITTED  = 12.52   # diagnose_synloc_scale.py, contaminated global fit

    print(f"\n  {'Metric':<12}  {'Nominal (eval)':>16}  {'Fitted (scale diag)':>20}  "
          f"{'Authoritative':>15}")
    print(f"  {'──'*20}")
    for label, pct in [("Median", 50), ("Mean", None), ("p90", 90), ("p95", 95)]:
        auth_v = (float(np.nanmedian(arr)) if pct == 50 else
                  float(np.nanmean(arr))   if pct is None else
                  float(np.nanpercentile(arr, pct)))
        nom_v  = (PREV_NOMINAL if pct == 50 else "—" if pct is None else "—")
        fit_v  = (PREV_FITTED  if pct == 50 else "—" if pct is None else "—")
        nom_s  = f"{nom_v:.2f} m" if isinstance(nom_v, float) else nom_v
        fit_s  = f"{fit_v:.2f} m" if isinstance(fit_v, float) else fit_v
        print(f"  {label:<12}  {nom_s:>16}  {fit_s:>20}  {auth_v:>13.2f} m")

    print(f"  n players: {len(arr)}")

    auth_med = float(np.nanmedian(arr))
    delta_vs_nominal = PREV_NOMINAL - auth_med
    print(f"\n  Improvement vs nominal: {delta_vs_nominal:+.2f} m  "
          f"({100*delta_vs_nominal/PREV_NOMINAL:.1f}% reduction in median)")

    return auth_med


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Good-calibration frames only
# ════════════════════════════════════════════════════════════════════════════════

def step3_good_cal_accuracy(img_results, sign_len, sign_wid):
    print()
    print("=" * 72)
    print("  STEP 3 — Good-calibration frames (intrinsic ≤ 5 m)")
    print("=" * 72)
    print(f"\n  Split: intrinsic ≤ {GOOD_CAL_INTRINSIC_M} m = 'good calibration'  |  "
          f"> {GOOD_CAL_INTRINSIC_M} m or None = 'calibration failure'")

    good_cal = [r for r in img_results
                if r["intrinsic_median_m"] is not None
                and r["intrinsic_median_m"] <= GOOD_CAL_INTRINSIC_M]
    cal_fail = [r for r in img_results
                if r["intrinsic_median_m"] is None
                or r["intrinsic_median_m"] > GOOD_CAL_INTRINSIC_M]

    print(f"  Good calibration: {len(good_cal)} frames  |  "
          f"Calibration failure: {len(cal_fail)} frames")

    # Good-cal aggregate
    gc_errs = []
    for r in good_cal:
        gc_errs.extend(r["auth_errs"])
    gc_arr = np.array(gc_errs)

    print(f"\n  ── Position error, good-calibration frames only ─────────────────")
    print(f"  Median : {np.nanmedian(gc_arr):.2f} m   ← primary accuracy number")
    print(f"  Mean   : {np.nanmean(gc_arr):.2f} m")
    print(f"  p90    : {np.nanpercentile(gc_arr, 90):.2f} m")
    print(f"  p95    : {np.nanpercentile(gc_arr, 95):.2f} m")
    print(f"  n      : {len(gc_arr)} players across {len(good_cal)} frames")

    # Flag frames with per-frame median > FLIP_POSITION_M → dihedral flip
    ok_frames   = [r for r in good_cal if r["auth_median_err"] <= FLIP_POSITION_M]
    flip_frames = [r for r in good_cal if r["auth_median_err"] >  FLIP_POSITION_M]

    print(f"\n  ── Good-cal frame classification (threshold {FLIP_POSITION_M} m) ──────────")
    print(f"  Correct pose (≤ {FLIP_POSITION_M} m)  : {len(ok_frames)} frames")
    print(f"  Flipped pose (> {FLIP_POSITION_M} m)  : {len(flip_frames)} frames")

    if flip_frames:
        print(f"\n  Flagged as dihedral/pose flip (correct intrinsics, wrong pitch end):")
        print(f"  {'File':<20}  {'intrinsic_m':>12}  {'auth_pos_m':>11}  {'n_kp':>5}")
        print(f"  {'──'*18}")
        for r in sorted(flip_frames, key=lambda r: r["auth_median_err"], reverse=True):
            print(f"  {r['file_name']:<20}  {r['intrinsic_median_m']:>12.2f}  "
                  f"{r['auth_median_err']:>11.2f}  {r['n_kp']:>5}")

    # Only-correct-pose frames accuracy
    if ok_frames:
        op_errs = []
        for r in ok_frames:
            op_errs.extend(r["auth_errs"])
        op_arr = np.array(op_errs)
        print(f"\n  ── Position error, correct-pose frames only ({len(ok_frames)} frames) ──────")
        print(f"  Median : {np.nanmedian(op_arr):.2f} m")
        print(f"  Mean   : {np.nanmean(op_arr):.2f} m")
        print(f"  p90    : {np.nanpercentile(op_arr, 90):.2f} m")
        print(f"  n      : {len(op_arr)} players")

    return ok_frames, flip_frames, cal_fail


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Three-way breakdown
# ════════════════════════════════════════════════════════════════════════════════

def step4_breakdown(ok_frames, flip_frames, cal_fail, eval_data):
    print()
    print("=" * 72)
    print("  STEP 4 — Complete breakdown of the 50-image run")
    print("=" * 72)

    n_total      = len(eval_data["per_image"])
    n_never      = sum(1 for r in eval_data["per_image"] if r["status"] == "failed")
    n_calibrated = n_total - n_never  # 29

    n_ok   = len(ok_frames)
    n_flip = len(flip_frames)
    n_fail = len(cal_fail)
    # sanity check
    assert n_ok + n_flip + n_fail == n_calibrated, (
        f"counts don't add up: {n_ok}+{n_flip}+{n_fail} ≠ {n_calibrated}")

    def pct(n): return 100 * n / n_total

    print(f"\n  Total images in run: {n_total}")
    print()
    print(f"  ── Completion ──────────────────────────────────────────────────")
    print(f"  Never calibrated (kp < 4 or PnLCalib fail)  : "
          f"{n_never:>3}  ({pct(n_never):>5.1f}%)")
    print(f"  Calibrated (kp ≥ 4, H recovered)            : "
          f"{n_calibrated:>3}  ({pct(n_calibrated):>5.1f}%)")
    print()
    print(f"  ── Among calibrated frames ─────────────────────────────────────")
    print(f"  (a) Good calibration + correct pose:         "
          f"{n_ok:>3}  ({pct(n_ok):>5.1f}% of total, "
          f"{100*n_ok/n_calibrated:.0f}% of calibrated)")
    print(f"       → intrinsic ≤ {GOOD_CAL_INTRINSIC_M} m AND pos ≤ {FLIP_POSITION_M} m")
    print(f"       → these produce usable player positions")
    print()
    print(f"  (b) Good calibration, flipped pose:          "
          f"{n_flip:>3}  ({pct(n_flip):>5.1f}% of total, "
          f"{100*n_flip/n_calibrated:.0f}% of calibrated)")
    print(f"       → intrinsic ≤ {GOOD_CAL_INTRINSIC_M} m BUT pos > {FLIP_POSITION_M} m")
    print(f"       → PnLCalib found the correct H geometry but for the wrong")
    print(f"          pitch orientation (dihedral ambiguity on symmetric views)")
    print()
    print(f"  (c) Calibration failure:                     "
          f"{n_fail:>3}  ({pct(n_fail):>5.1f}% of total, "
          f"{100*n_fail/n_calibrated:.0f}% of calibrated)")
    print(f"       → intrinsic > {GOOD_CAL_INTRINSIC_M} m or intrinsic not computable")
    print(f"       → PnLCalib found a geometrically wrong H")
    print()
    print(f"  ── Summary ─────────────────────────────────────────────────────")

    def bar(n, total, width=30):
        filled = int(round(n / total * width))
        return "█" * filled + "░" * (width - filled)

    for label, n in [
        ("Never cal",    n_never),
        ("(a) Working",  n_ok),
        ("(b) Flipped",  n_flip),
        ("(c) Cal fail", n_fail),
    ]:
        print(f"  {label:<12}  {bar(n, n_total)}  {n:>2} / {n_total}  "
              f"({pct(n):.0f}%)")

    print()
    # What fraction of calibrated images produce usable output?
    print(f"  Usable calibration rate (a): {n_ok}/{n_calibrated} = "
          f"{100*n_ok/n_calibrated:.0f}% of calibrated")
    print(f"  Overall usable rate (a/total): {n_ok}/{n_total} = "
          f"{100*n_ok/n_total:.0f}% of attempted images")
    if n_flip > 0:
        print(f"  Recoverable with orientation fix (a+b): "
              f"{n_ok+n_flip}/{n_calibrated} = "
              f"{100*(n_ok+n_flip)/n_calibrated:.0f}% of calibrated")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Apply authoritative SynLoc transform; measure position error.")
    ap.add_argument("--eval-json", default=DEFAULT_EVAL_JSON)
    ap.add_argument("--ann-file",  default=VAL_ANN_FILE)
    ap.add_argument("--val-dir",   default=VAL_IMG_DIR)
    args = ap.parse_args()
    args.eval_json = os.path.abspath(args.eval_json)
    args.ann_file  = os.path.abspath(args.ann_file)
    args.val_dir   = os.path.abspath(args.val_dir)

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

    n_cal = sum(1 for r in eval_data["per_image"] if r["status"] == "calibrated")
    print(f"Run summary: {len(eval_data['per_image'])} total, {n_cal} calibrated")

    print("\nLoading PnLCalib models...")
    model_kp, model_l = load_models()
    print("Models loaded.")
    print()

    img_results = recalibrate_all(
        eval_data, images_by_id, anns_by_img, model_kp, model_l)

    best_combo, sign_len, sign_wid = step1_resolve_signs(img_results)
    step2_aggregate(img_results, sign_len, sign_wid)
    ok_frames, flip_frames, cal_fail = step3_good_cal_accuracy(img_results, sign_len, sign_wid)
    step4_breakdown(ok_frames, flip_frames, cal_fail, eval_data)

    print()
    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
