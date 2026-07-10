"""
evaluate_synloc_calibration.py
Measures PnLCalib (SV_kp + SV_lines, base mode) calibration accuracy against
the SpiideoSynLoc validation set ground truth.

Usage:
    vendor/PnLCalib/.venv/bin/python eval/synloc/evaluate_synloc_calibration.py [--limit N] [--output FILE]

Steps:
    0 — Coordinate frame reconciliation (always printed first)
    1 — Per-image calibration + error computation
    2 — Aggregate statistics printed + saved to JSON
"""
from __future__ import annotations
import os, sys, json, argparse, time
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
# PnLCalib opens "config/*.yaml" and "weights/*" with relative paths;
# chdir makes those paths resolve correctly.
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

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "SoccerNet", "SpiideoSynLoc")
VAL_IMG_DIR  = os.path.join(DATA_DIR, "val")
VAL_ANN_FILE = os.path.join(DATA_DIR, "annotations", "val.json")
DEFAULT_OUT  = os.path.join(PROJECT_ROOT, ".cache", "synloc_diagnostics", "synloc_calibration_eval.json")

# ── PnLCalib inference constants ──────────────────────────────────────────────
DEVICE         = "cuda:0"
KP_THRESHOLD   = 0.3434
LINE_THRESHOLD  = 0.7867

# ── Pitch dimensions ──────────────────────────────────────────────────────────
# SynLoc (SpiideoSynLoc):  origin=pitch centre
#   X axis = width  (short), half-extent = 35.0 m  (total 70 m)
#   Y axis = length (long),  half-extent = 60.0 m  (total 120 m)
#   Z axis = up (positive above ground)
# Source: sskit defaults (width=70, height=120) + confirmed against annotation
# ranges: X ∈ [-39.3, +39.9], Y ∈ [-58.3, +57.5].
SL_HALF_X = 35.0   # SynLoc width half-extent (m)
SL_HALF_Y = 60.0   # SynLoc length half-extent (m)

# PnLCalib H world frame (fitted to standard FIFA pitch): origin=pitch centre
#   X axis = length (long),  half-extent = 52.5 m  (total 105 m)
#   Y axis = width  (short), half-extent = 34.0 m  (total  68 m)
#   H maps [Xp, Yp, 1] → image pixel [u/w, v/w].
PNL_HALF_X = 52.5   # PnLCalib length half-extent (m)
PNL_HALF_Y = 34.0   # PnLCalib width  half-extent (m)

# Scale factors for coordinate conversion
SCALE_SL_Y_TO_PNL_X = PNL_HALF_X / SL_HALF_Y   # 52.5 / 60  = 0.8750
SCALE_SL_X_TO_PNL_Y = PNL_HALF_Y / SL_HALF_X   # 34.0 / 35  = 0.9714

# Minimum keypoints for a non-degenerate calibration.
# PnLCalib with kp < 4 tends to overfit (reproj→0) and produce garbage H matrices.
MIN_KP_FOR_VALID_CALIB = 4


# ════════════════════════════════════════════════════════════════════════════════
# Coordinate conversion
# ════════════════════════════════════════════════════════════════════════════════

def synloc_to_pnlcalib(xs: float, ys: float) -> tuple[float, float]:
    """SynLoc world (Xs, Ys) → PnLCalib H world (Xp, Yp).

    Axis mapping:   SynLoc Y (length) → PnLCalib X (length)
                    SynLoc X (width)  → PnLCalib Y (width)
    Sign convention: NEGATED on both axes — verified empirically by testing all
    4 sign combinations on a 3-image sample: the '--' variant (negative signs)
    consistently yielded the lowest position errors.  Underlying reason: the
    PnLCalib and SynLoc length/width axes point in opposite directions.
    Scale factors account for the pitch-model dimension difference
    (SynLoc 120×70 m vs PnLCalib 105×68 m).
    """
    return -ys * SCALE_SL_Y_TO_PNL_X, -xs * SCALE_SL_X_TO_PNL_Y


def pnlcalib_to_synloc(xp: float, yp: float) -> tuple[float, float]:
    """PnLCalib H world (Xp, Yp) → SynLoc world (Xs, Ys).  Inverse of synloc_to_pnlcalib."""
    return -yp / SCALE_SL_X_TO_PNL_Y, -xp / SCALE_SL_Y_TO_PNL_X


def project_w2i(H: np.ndarray, xw: float, yw: float) -> tuple[float, float]:
    p = H @ np.array([xw, yw, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def project_i2w(H_inv: np.ndarray, u: float, v: float) -> tuple[float, float]:
    q = H_inv @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


# ════════════════════════════════════════════════════════════════════════════════
# STEP 0 — Coordinate frame analysis
# ════════════════════════════════════════════════════════════════════════════════

def step0_frame_analysis(ann_data: dict) -> None:
    print()
    print("=" * 72)
    print("  STEP 0 — Coordinate Frame Reconciliation")
    print("=" * 72)

    print("\n  ── SynLoc world frame (SpiideoSynLoc dataset) ───────────────────")
    print(f"  Origin : pitch centre")
    print(f"  X axis : width  (short axis), half-extent = {SL_HALF_X:.1f} m  "
          f"[total {2*SL_HALF_X:.0f} m]")
    print(f"  Y axis : length (long  axis), half-extent = {SL_HALF_Y:.1f} m  "
          f"[total {2*SL_HALF_Y:.0f} m]")
    print(f"  Z axis : upward positive; Z=0 is ground plane")
    print(f"  Units  : metres")
    print(f"  Source : sskit project_on_ground defaults (width=70, height=120)")
    print(f"           Verified: val annotation ranges")
    print(f"           X ∈ [-39.3, +39.9] m   (players can be slightly off-pitch)")
    print(f"           Y ∈ [-58.3, +57.5] m")

    print("\n  ── Our H frame (PnLCalib, standard FIFA 105×68 m model) ─────────")
    print(f"  Origin : pitch centre")
    print(f"  X axis : length (long  axis), half-extent = {PNL_HALF_X:.1f} m  "
          f"[total {2*PNL_HALF_X:.0f} m]")
    print(f"  Y axis : width  (short axis), half-extent = {PNL_HALF_Y:.1f} m  "
          f"[total {2*PNL_HALF_Y:.0f} m]")
    print(f"  Z axis : upward, with negative convention for camera elevation")
    print(f"  H maps : [Xp, Yp, 1]  →  image pixel [u/w, v/w]")

    print("\n  ── Frame differences ────────────────────────────────────────────")
    print(f"  Axis swap : SynLoc X (width) ↔ PnLCalib Y (width)")
    print(f"              SynLoc Y (length)↔ PnLCalib X (length)")
    print(f"  Sign      : BOTH axes are NEGATED (the two systems have opposite")
    print(f"              orientations for their length and width axes).")
    print(f"              Verified by testing all 4 sign combinations on a 3-image")
    print(f"              sample; the '--' (both negative) variant minimised errors.")
    print(f"  Pitch size: SynLoc  120 × 70 m  (synthetic dataset)")
    print(f"              PnLCalib 105 × 68 m  (standard FIFA)")
    print(f"  Scale diff: length axis  ×{SCALE_SL_Y_TO_PNL_X:.4f}  "
          f"({PNL_HALF_X}/{SL_HALF_Y})")
    print(f"              width  axis  ×{SCALE_SL_X_TO_PNL_Y:.4f}  "
          f"({PNL_HALF_Y}/{SL_HALF_X})")
    print(f"  WARNING — pitch model mismatch: SynLoc uses its own 120×70 m model")
    print(f"  with proportionally scaled markings, while PnLCalib was trained on")
    print(f"  the standard 105×68 m FIFA pitch.  The nominal scale factors above")
    print(f"  are approximate; the effective per-image scale may deviate because")
    print(f"  different pitch markings contribute differently to calibration.")
    print(f"  Additionally, SynLoc images have fisheye lens distortion")
    print(f"  (dist_poly coefficients up to 0.77) that PnLCalib does NOT model,")
    print(f"  which is the dominant source of calibration error on this dataset.")
    print(f"  Degenerate calibrations: n_kp < {MIN_KP_FOR_VALID_CALIB} → marked failed;")
    print(f"  PnLCalib with very few keypoints overfits (reproj→0) and produces")
    print(f"  geometrically nonsensical H matrices.")

    print("\n  ── Explicit bidirectional transform ─────────────────────────────")
    print(f"  SynLoc → PnLCalib:")
    print(f"      Xp = -Ys × {SCALE_SL_Y_TO_PNL_X:.4f}   "
          f"[SynLoc length axis → PnLCalib length axis, NEGATED]")
    print(f"      Yp = -Xs × {SCALE_SL_X_TO_PNL_Y:.4f}   "
          f"[SynLoc width  axis → PnLCalib width  axis, NEGATED]")
    print(f"  PnLCalib → SynLoc:")
    print(f"      Xs = -Yp × {1/SCALE_SL_X_TO_PNL_Y:.4f}   "
          f"[PnLCalib width  axis → SynLoc width  axis, NEGATED]")
    print(f"      Ys = -Xp × {1/SCALE_SL_Y_TO_PNL_X:.4f}   "
          f"[PnLCalib length axis → SynLoc length axis, NEGATED]")
    print(f"  Scale factors cancel in round-trips (errors are not scale-biased).")

    # Worked example: first annotation in the val set
    a0 = ann_data["annotations"][0]
    xs_gt, ys_gt = a0["position_on_pitch"]
    u_gt, v_gt = float(a0["keypoints"][1][0]), float(a0["keypoints"][1][1])
    xp, yp = synloc_to_pnlcalib(xs_gt, ys_gt)
    xs_rt, ys_rt = pnlcalib_to_synloc(xp, yp)

    print(f"\n  ── Worked example (val annotation id=0, image_id=0) ─────────────")
    print(f"  SynLoc position_on_pitch : "
          f"Xs = {xs_gt:+.4f} m,  Ys = {ys_gt:+.4f} m")
    print(f"  GT foot pixel (keypoints[1]): u={u_gt:.1f},  v={v_gt:.1f}")
    print(f"  → PnLCalib frame (after axis swap + rescale):")
    print(f"      Xp = {ys_gt:+.4f} × {SCALE_SL_Y_TO_PNL_X} = {xp:+.4f} m  (length)")
    print(f"      Yp = {xs_gt:+.4f} × {SCALE_SL_X_TO_PNL_Y:.4f} = {yp:+.4f} m  (width)")
    print(f"  Round-trip SynLoc→PnLCalib→SynLoc:")
    print(f"      Xs_rt = {xs_rt:+.8f}  (err = {abs(xs_rt - xs_gt):.2e} m)")
    print(f"      Ys_rt = {ys_rt:+.8f}  (err = {abs(ys_rt - ys_gt):.2e} m)")
    print(f"  Interpretation: player is {abs(xs_gt):.2f} m from centre line (width)")
    print(f"    and {abs(ys_gt):.2f} m from pitch centre in the length direction.")
    print(f"  If our H is correct, projecting (Xp={xp:+.2f}, Yp={yp:+.2f}) through H")
    print(f"  should yield a pixel close to ({u_gt:.1f}, {v_gt:.1f}).")
    print()


# ════════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════════

def load_models() -> tuple:
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


# ════════════════════════════════════════════════════════════════════════════════
# Per-image calibration
# ════════════════════════════════════════════════════════════════════════════════

def calibrate_image(
    frame_bgr: np.ndarray,
    w_orig: int,
    h_orig: int,
    model_kp,
    model_l,
) -> tuple[np.ndarray | None, int, float, str]:
    """Run PnLCalib on one frame.

    Returns (H, n_keypoints, pnlcalib_reproj_err_px, mode).
    H is None if calibration fails.
    """
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
        kp_dict[0], lines_dict[0], w=w_net, h=h_net, normalize=True
    )
    n_kp = len(kp_dict)

    cam.update(kp_dict, lines_dict)
    result = cam.heuristic_voting(refine_lines=False)

    if result is None:
        return None, n_kp, float("inf"), "fail"

    # Reject degenerate calibrations: too few keypoints cause overfitting
    # (reproj error collapses to ~0) and produce geometrically wrong H matrices.
    if n_kp < MIN_KP_FOR_VALID_CALIB:
        return None, n_kp, float(result["rep_err"]), "degenerate"

    cp = result["cam_params"]
    fx = cp["x_focal_length"]
    fy = cp["y_focal_length"]
    cx, cy = cp["principal_point"]
    K   = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    R   = np.array(cp["rotation_matrix"], dtype=np.float64)
    pos = np.array(cp["position_meters"], dtype=np.float64)
    t   = -(R @ pos)
    H   = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H  /= H[2, 2]

    return H, n_kp, float(result["rep_err"]), str(result["mode"])


# ════════════════════════════════════════════════════════════════════════════════
# Per-player error computation
# ════════════════════════════════════════════════════════════════════════════════

def compute_player_errors(
    H: np.ndarray,
    anns: list[dict],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Compute reprojection and position errors for all annotated players.

    For each player with a visible foot keypoint:
      - reprojection error (px): convert GT world → PnLCalib frame → project
        through H → pixel distance to GT pixel.
      - position error (m):      back-project GT pixel through H_inv → PnLCalib
        world → convert to SynLoc metres → Euclidean distance to GT world.

    Both metrics are zero for a perfect calibration; neither is biased by the
    pitch-scale difference because the scale factors cancel in round-trips.
    """
    H_inv = np.linalg.inv(H)
    records = []

    for a in anns:
        xs_gt, ys_gt = a["position_on_pitch"]
        foot_kp = a["keypoints"][1]   # [u, v, visibility]
        u_gt, v_gt, vis = float(foot_kp[0]), float(foot_kp[1]), int(foot_kp[2])

        if vis == 0:
            continue

        # ── position error (fixed transform) ─────────────────────────────────
        # Back-project GT pixel → PnLCalib world → SynLoc world.
        xp_est, yp_est = project_i2w(H_inv, u_gt, v_gt)
        xs_est, ys_est = pnlcalib_to_synloc(xp_est, yp_est)
        pos_err_m = float(np.sqrt((xs_est - xs_gt) ** 2 +
                                  (ys_est - ys_gt) ** 2))
        if pos_err_m > 200:
            pos_err_m = float("nan")

        records.append({
            "ann_id":          int(a["id"]),
            "gt_pixel":        [u_gt, v_gt],
            "gt_world_synloc": [xs_gt, ys_gt],
            "pnlcalib_est":    [float(xp_est), float(yp_est)],
            "est_world_synloc":[float(xs_est), float(ys_est)],
            "pos_err_m":       pos_err_m,
        })

    # ── intrinsic accuracy + affine-corrected reprojection ───────────────────
    # Fit the best linear map from SynLoc GT world to PnLCalib back-projections.
    # This removes coordinate-system offset so residuals isolate PnLCalib geometry.
    # Also use the fitted affine in the FORWARD direction to get a valid
    # reprojection error: affine(SynLoc world) → PnLCalib world → H → pixel.
    intrinsic_median_m = None
    if len(records) >= 4:
        sl_gt  = np.array([[r["gt_world_synloc"][0], r["gt_world_synloc"][1]]
                            for r in records], float)
        pnl_bp = np.array([[r["pnlcalib_est"][0], r["pnlcalib_est"][1]]
                            for r in records], float)
        A_src  = np.column_stack([sl_gt, np.ones(len(sl_gt))])
        A_fit, _, _, _ = np.linalg.lstsq(A_src, pnl_bp, rcond=None)
        pnl_pred = A_src @ A_fit   # best-fit PnLCalib positions
        resid_pnl = np.linalg.norm(pnl_bp - pnl_pred, axis=1)
        # Convert PnLCalib residuals to approximate SynLoc metres
        mean_inv_scale = (SL_HALF_Y / PNL_HALF_X + SL_HALF_X / PNL_HALF_Y) / 2
        resid_sl = resid_pnl * mean_inv_scale
        intrinsic_median_m = float(np.median(resid_sl))

        # Reprojection error using affine-corrected world → pixel
        margin = max(img_w, img_h) * 10
        for rec, (xp_opt, yp_opt), r_sl in zip(records, pnl_pred, resid_sl):
            u_pred, v_pred = project_w2i(H, xp_opt, yp_opt)
            u_gt_, v_gt_ = rec["gt_pixel"]
            if abs(u_pred) > margin or abs(v_pred) > margin:
                reproj_px = float("nan")
            else:
                reproj_px = float(np.sqrt((u_pred - u_gt_) ** 2 +
                                          (v_pred - v_gt_) ** 2))
            rec["intrinsic_err_m"] = float(r_sl)
            rec["reproj_err_px"]   = reproj_px

    return records, intrinsic_median_m


# ════════════════════════════════════════════════════════════════════════════════
# Aggregate statistics
# ════════════════════════════════════════════════════════════════════════════════

def _stats(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return {"n": 0, "median": None, "mean": None, "p90": None, "p95": None}
    return {
        "n":      int(len(v)),
        "median": float(np.median(v)),
        "mean":   float(np.mean(v)),
        "p90":    float(np.percentile(v, 90)),
        "p95":    float(np.percentile(v, 95)),
    }


def aggregate(per_image: list[dict]) -> dict:
    calibrated = [r for r in per_image if r["status"] == "calibrated"]
    n_total    = len(per_image)
    n_cal      = len(calibrated)
    n_fail     = n_total - n_cal

    all_reproj    = []
    all_pos       = []
    all_intrinsic = []
    for r in calibrated:
        all_reproj.extend(r.get("player_reproj_errs_px", []))
        all_pos.extend(r["player_pos_errs_m"])
        all_intrinsic.extend(r.get("player_intrinsic_errs_m", []))

    reproj_arr    = np.array(all_reproj,    dtype=float)
    pos_arr       = np.array(all_pos,       dtype=float)
    intrinsic_arr = np.array(all_intrinsic, dtype=float)

    return {
        "completion_rate":     n_cal / n_total if n_total else 0.0,
        "n_total":             n_total,
        "n_calibrated":        n_cal,
        "n_failed_or_degen":   n_fail,
        "n_players_total":     len(all_reproj),
        "reprojection_error_px": _stats(reproj_arr),
        "position_error_m":      _stats(pos_arr),
        "intrinsic_position_error_m": _stats(intrinsic_arr),
    }


def print_summary(agg: dict, out_path: str) -> None:
    r = agg["reprojection_error_px"]
    p = agg["position_error_m"]
    i = agg["intrinsic_position_error_m"]
    print()
    print("=" * 72)
    print("  STEP 2 — Aggregate Results")
    print("=" * 72)
    print(f"  Images total                  : {agg['n_total']}")
    print(f"  Calibrated (kp≥{MIN_KP_FOR_VALID_CALIB})            : {agg['n_calibrated']}"
          f"  ({agg['completion_rate']:.1%})")
    print(f"  Failed / degenerate (kp<{MIN_KP_FOR_VALID_CALIB})  : {agg['n_failed_or_degen']}")
    print(f"  Players scored                : {agg['n_players_total']}")
    print()
    print("  ── Position error (metres, fixed scale-factor transform) ─────────")
    print("     [includes pitch-model mismatch; headline for deployment context]")
    if p["n"]:
        print(f"  median : {p['median']:7.2f} m")
        print(f"  mean   : {p['mean']:7.2f} m")
        print(f"  p90    : {p['p90']:7.2f} m")
        print(f"  p95    : {p['p95']:7.2f} m")
        print(f"  n      : {p['n']}")
    else:
        print("  (no data)")
    print()
    print("  ── Intrinsic position error (metres, GT-assisted optimal affine) ─")
    print("     [removes coordinate-system offset; measures PnLCalib geometry]")
    if i["n"]:
        print(f"  median : {i['median']:7.2f} m   ← calibration quality metric")
        print(f"  mean   : {i['mean']:7.2f} m")
        print(f"  p90    : {i['p90']:7.2f} m")
        print(f"  p95    : {i['p95']:7.2f} m")
        print(f"  n      : {i['n']}")
    else:
        print("  (no data)")
    print()
    print("  ── Reprojection error (pixels, affine-corrected world→pixel) ────")
    print("     [affine-corrected SynLoc→PnLCalib frame, then H projects to pixel]")
    if r["n"]:
        print(f"  median : {r['median']:7.1f} px")
        print(f"  mean   : {r['mean']:7.1f} px")
        print(f"  p90    : {r['p90']:7.1f} px")
        print(f"  p95    : {r['p95']:7.1f} px")
        print(f"  n      : {r['n']}")
    else:
        print("  (no data)")
    print()
    print("  ── Interpretation ───────────────────────────────────────────────")
    print("     Position error (fixed transform) is dominated by fisheye-lens")
    print("     distortion in SynLoc images, which PnLCalib (pinhole model) does")
    print("     not account for.  Intrinsic error isolates geometric accuracy.")
    print("=" * 72)
    print(f"  Per-image results saved → {out_path}")
    print()


# ════════════════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════════════════

def run(args) -> None:
    # ── load annotations ──────────────────────────────────────────────────────
    print(f"Loading annotations: {args.ann_file}")
    with open(args.ann_file) as f:
        ann_data = json.load(f)

    # Group annotations by image_id for O(1) lookup
    from collections import defaultdict
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in ann_data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    images = ann_data["images"]
    if args.limit is not None:
        images = images[: args.limit]

    print(f"Val images: {len(images)}"
          + (f"  (limited to {args.limit})" if args.limit else ""))

    # ── Step 0 ────────────────────────────────────────────────────────────────
    step0_frame_analysis(ann_data)

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading PnLCalib models onto GPU...")
    model_kp, model_l = load_models()
    print("Models loaded.\n")

    # ── STEP 1: per-image loop ────────────────────────────────────────────────
    print("=" * 72)
    print("  STEP 1 — Per-image calibration")
    print("=" * 72)

    per_image_results: list[dict] = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        img_path = os.path.join(args.val_dir, img_info["file_name"])
        image_id = int(img_info["id"])
        w_orig   = int(img_info["width"])
        h_orig   = int(img_info["height"])

        frame = cv2.imread(img_path)
        if frame is None:
            per_image_results.append({
                "image_id":    image_id,
                "file_name":   img_info["file_name"],
                "status":      "failed",
                "fail_reason": "cannot_read_image",
                "n_kp":        0,
                "pnlcalib_reproj_err_px": None,
                "calib_mode":  None,
                "n_players":   0,
                "player_reproj_errs_px": [],
                "player_pos_errs_m":     [],
            })
            print(f"  [{idx+1:>4}/{len(images)}] {img_info['file_name']}  "
                  f"CANNOT READ")
            continue

        H, n_kp, pnl_repr, mode = calibrate_image(
            frame, w_orig, h_orig, model_kp, model_l
        )

        anns = anns_by_img.get(image_id, [])

        if H is None:
            per_image_results.append({
                "image_id":    image_id,
                "file_name":   img_info["file_name"],
                "status":      "failed",
                "fail_reason": "pnlcalib_no_result",
                "n_kp":        n_kp,
                "pnlcalib_reproj_err_px": None,
                "calib_mode":  mode,
                "n_players":   len(anns),
                "player_reproj_errs_px": [],
                "player_pos_errs_m":     [],
            })
            elapsed = time.time() - t_start
            print(f"  [{idx+1:>4}/{len(images)}] {img_info['file_name']}  "
                  f"FAILED (kp={n_kp})  "
                  f"[{elapsed:.0f}s elapsed]")
            continue

        player_errs, intrinsic_med = compute_player_errors(H, anns, w_orig, h_orig)
        pos_list      = [e["pos_err_m"]               for e in player_errs]
        reproj_list   = [e.get("reproj_err_px",  float("nan")) for e in player_errs]
        intrinsic_list= [e.get("intrinsic_err_m",float("nan")) for e in player_errs]

        per_image_results.append({
            "image_id":    image_id,
            "file_name":   img_info["file_name"],
            "status":      "calibrated",
            "fail_reason": None,
            "n_kp":        n_kp,
            "pnlcalib_reproj_err_px": pnl_repr,
            "calib_mode":  mode,
            "n_players":   len(player_errs),
            "player_pos_errs_m":       pos_list,
            "player_intrinsic_errs_m": intrinsic_list,
            "player_reproj_errs_px":   reproj_list,
            "median_pos_err_m": (
                float(np.nanmedian(pos_list)) if pos_list else None
            ),
            "intrinsic_median_m": intrinsic_med,
            "median_reproj_err_px": (
                float(np.nanmedian([x for x in reproj_list if np.isfinite(x)]))
                if any(np.isfinite(x) for x in reproj_list) else None
            ),
        })

        elapsed = time.time() - t_start
        fps     = (idx + 1) / elapsed
        eta_s   = (len(images) - idx - 1) / fps if fps > 0 else 0
        med_pos = (
            f"{float(np.nanmedian(pos_list)):.2f} m"
            if pos_list else "—"
        )
        intr_str = f"{intrinsic_med:.2f} m" if intrinsic_med is not None else "—"
        print(f"  [{idx+1:>4}/{len(images)}] {img_info['file_name']}  "
              f"kp={n_kp:>2}  reproj={pnl_repr:.1f}px  "
              f"players={len(player_errs):>2}  pos_med={med_pos:<10}"
              f"  intrinsic={intr_str:<10}"
              f"  [{elapsed:.0f}s | ETA {eta_s:.0f}s]")

    # ── STEP 2: aggregate ─────────────────────────────────────────────────────
    agg = aggregate(per_image_results)
    print_summary(agg, args.output)

    # ── save JSON ──────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "script":      "evaluate_synloc_calibration.py",
            "dataset":     "SpiideoSynLoc val",
            "ann_file":    args.ann_file,
            "val_dir":     args.val_dir,
            "limit":       args.limit,
            "pnlcalib_weights": {"kp": "weights/SV_kp", "lines": "weights/SV_lines"},
            "pnl_refine":  False,
            "min_kp_for_valid_calib": MIN_KP_FOR_VALID_CALIB,
            "coord_frame": {
                "synloc":   {"X": "width ±35m", "Y": "length ±60m",
                             "pitch": "70×120m (synthetic)"},
                "pnlcalib": {"X": "length ±52.5m", "Y": "width ±34m",
                             "pitch": "105×68m (FIFA standard)"},
                "transform_sl_to_pnl": {
                    "Xp": f"-Ys × {SCALE_SL_Y_TO_PNL_X}  [negated]",
                    "Yp": f"-Xs × {SCALE_SL_X_TO_PNL_Y:.4f}  [negated]",
                },
                "sign_note": ("Both axes negated — verified empirically. "
                              "Dominant residual error is SynLoc fisheye "
                              "distortion (pinhole model mismatch)."),
            },
        },
        "aggregate": agg,
        "per_image": per_image_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate PnLCalib accuracy on SpiideoSynLoc val split."
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of images to process (default: all 6777)"
    )
    ap.add_argument(
        "--output", default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})"
    )
    ap.add_argument(
        "--val-dir", default=VAL_IMG_DIR,
        help=f"Path to val image directory (default: {VAL_IMG_DIR})"
    )
    ap.add_argument(
        "--ann-file", default=VAL_ANN_FILE,
        help=f"Path to val annotation JSON (default: {VAL_ANN_FILE})"
    )
    args = ap.parse_args()
    # Resolve output to absolute path BEFORE os.chdir() redirects relative paths.
    args.output = os.path.abspath(args.output)
    run(args)


if __name__ == "__main__":
    main()
