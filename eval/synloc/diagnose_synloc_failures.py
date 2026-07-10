"""
diagnose_synloc_failures.py
Diagnostic on the existing 50-image synloc_calibration_eval.json run.

Does NOT re-run the full evaluation. Only re-calibrates the N worst images
(default 6) to obtain H matrices needed for Check 1 transform classification.

Run with:
    vendor/PnLCalib/.venv/bin/python eval/synloc/diagnose_synloc_failures.py [--eval-json FILE] [--out-dir DIR]

Outputs:
    diag_worst_<0-5>.png          — side-by-side GT vs estimated top-down plots
    diag_radial_vs_error.png      — scatter: radial distance vs position error
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
# Must chdir before importing PnLCalib — it opens config/*.yaml with relative paths.
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
DATA_DIR          = os.path.join(PROJECT_ROOT, "data", "SoccerNet", "SpiideoSynLoc")
VAL_IMG_DIR       = os.path.join(DATA_DIR, "val")
VAL_ANN_FILE      = os.path.join(DATA_DIR, "annotations", "val.json")
DEFAULT_EVAL_JSON = os.path.join(PROJECT_ROOT, ".cache", "synloc_diagnostics", "synloc_calibration_eval.json")
DEFAULT_OUT_DIR   = os.path.join(PROJECT_ROOT, ".cache", "synloc_diagnostics")

# ── Inference settings ────────────────────────────────────────────────────────
DEVICE                 = "cuda:0"
KP_THRESHOLD           = 0.3434
LINE_THRESHOLD         = 0.7867
MIN_KP_FOR_VALID_CALIB = 4

# ── Coordinate system constants ───────────────────────────────────────────────
SL_HALF_X          = 35.0    # SynLoc width  half-extent (m)
SL_HALF_Y          = 60.0    # SynLoc length half-extent (m)
PNL_HALF_X         = 52.5   # PnLCalib length half-extent (m)
PNL_HALF_Y         = 34.0   # PnLCalib width  half-extent (m)
SCALE_SL_Y_TO_PNL_X = PNL_HALF_X / SL_HALF_Y   # 0.8750
SCALE_SL_X_TO_PNL_Y = PNL_HALF_Y / SL_HALF_X   # 0.9714


def synloc_to_pnlcalib(xs: float, ys: float):
    return -ys * SCALE_SL_Y_TO_PNL_X, -xs * SCALE_SL_X_TO_PNL_Y


def pnlcalib_to_synloc(xp: float, yp: float):
    return -yp / SCALE_SL_X_TO_PNL_Y, -xp / SCALE_SL_Y_TO_PNL_X


def project_i2w(H_inv, u, v):
    q = H_inv @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


# ── Model loading ─────────────────────────────────────────────────────────────
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
    """Identical to evaluate_synloc_calibration.py — returns (H, n_kp, reproj_err, mode)."""
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
    if n_kp < MIN_KP_FOR_VALID_CALIB:
        return None, n_kp, float(result["rep_err"]), "degenerate"

    cp  = result["cam_params"]
    K   = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]],
                    [0, cp["y_focal_length"], cp["principal_point"][1]],
                    [0, 0, 1]], dtype=np.float64)
    R   = np.array(cp["rotation_matrix"], dtype=np.float64)
    pos = np.array(cp["position_meters"],  dtype=np.float64)
    t   = -(R @ pos)
    H   = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H  /= H[2, 2]
    return H, n_kp, float(result["rep_err"]), str(result["mode"])


# ── Pitch drawing ─────────────────────────────────────────────────────────────
def draw_pitch(ax, half_l=PNL_HALF_X, half_w=PNL_HALF_Y):
    ax.set_facecolor("#2a7a2a")
    ax.add_patch(patches.Rectangle(
        (-half_l, -half_w), 2 * half_l, 2 * half_w,
        linewidth=2, edgecolor="white", facecolor="none"))
    ax.plot([0, 0], [-half_w, half_w], "white", lw=1.2)
    ax.add_patch(plt.Circle((0, 0), 9.15, color="white", fill=False, lw=1.2))
    ax.plot(0, 0, "wo", ms=4)
    ax.set_xlim(-half_l - 6, half_l + 6)
    ax.set_ylim(-half_w - 6, half_w + 6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m, length axis)")
    ax.set_ylabel("Y (m, width axis)")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Transform classification
# ════════════════════════════════════════════════════════════════════════════════

# Each transform is applied to the ESTIMATED SynLoc position (xs_est, ys_est)
# obtained by H_inv-backprojecting GT pixels and converting back to SynLoc frame.
# We compare the transformed estimate to the GT SynLoc position.
# "identity" = the current result (no additional adjustment).
# If "identity" is best → genuine calibration failure.
# If a single other transform dominates → systematic convention bug.
TRANSFORMS = {
    "identity":    lambda xs, ys: (xs, ys),
    "negate_X":    lambda xs, ys: (-xs, ys),
    "negate_Y":    lambda xs, ys: (xs, -ys),
    "negate_both": lambda xs, ys: (-xs, -ys),
    "swap_XY":     lambda xs, ys: (ys, xs),
}


def check1(data, images_by_id, anns_by_img, model_kp, model_l, out_dir, n_worst):
    print()
    print("=" * 72)
    print("  CHECK 1 — Transform classification of worst-error frames")
    print("=" * 72)
    print(f"  (re-running PnLCalib on {n_worst} images to obtain H; no full re-eval)")

    calibrated = [r for r in data["per_image"]
                  if r["status"] == "calibrated" and r.get("median_pos_err_m") is not None]
    worst = sorted(calibrated, key=lambda r: r["median_pos_err_m"], reverse=True)[:n_worst]

    print(f"\n  Worst {n_worst} images by median position error:")
    for r in worst:
        print(f"    {r['file_name']:20s}  pos_err={r['median_pos_err_m']:6.1f}m  "
              f"intrinsic={r.get('intrinsic_median_m') or float('nan'):6.1f}m  kp={r['n_kp']}")

    frame_verdicts = {}

    for rank, img_record in enumerate(worst):
        image_id    = img_record["image_id"]
        fname       = img_record["file_name"]
        img_info    = images_by_id[image_id]
        img_path    = os.path.join(VAL_IMG_DIR, fname)
        w_orig      = int(img_info["width"])
        h_orig      = int(img_info["height"])

        print(f"\n  ── [{rank}] {fname}  (pos_err median={img_record['median_pos_err_m']:.1f}m) ──")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"    Cannot read image, skipping.")
            continue

        H, n_kp, reproj_err, mode = calibrate_image(frame, w_orig, h_orig, model_kp, model_l)
        print(f"    Re-calibrated: n_kp={n_kp}  reproj={reproj_err:.2f}px  mode={mode}")

        if H is None:
            print(f"    PnLCalib failed on re-run — cannot classify, skipping.")
            continue

        H_inv = np.linalg.inv(H)
        anns  = anns_by_img[image_id]

        # Build (gt_synloc, est_synloc, gt_pnlcalib, est_pnlcalib) tuples
        gt_sl_pts, est_sl_pts = [], []
        gt_pnl_pts, est_pnl_pts = [], []

        for a in anns:
            foot_kp = a["keypoints"][1]
            if int(foot_kp[2]) == 0:
                continue
            xs_gt = float(a["position_on_pitch"][0])
            ys_gt = float(a["position_on_pitch"][1])
            u_gt  = float(foot_kp[0])
            v_gt  = float(foot_kp[1])

            xp_gt, yp_gt   = synloc_to_pnlcalib(xs_gt, ys_gt)
            xp_est, yp_est = project_i2w(H_inv, u_gt, v_gt)
            xs_est, ys_est = pnlcalib_to_synloc(xp_est, yp_est)

            gt_sl_pts.append((xs_gt, ys_gt))
            est_sl_pts.append((xs_est, ys_est))
            gt_pnl_pts.append((xp_gt, yp_gt))
            est_pnl_pts.append((xp_est, yp_est))

        if len(gt_sl_pts) < 4:
            print(f"    Only {len(gt_sl_pts)} visible GT players — insufficient, skipping.")
            continue

        gt_sl_arr  = np.array(gt_sl_pts)
        est_sl_arr = np.array(est_sl_pts)

        # Test each transform on the estimated SynLoc positions
        print(f"    GT players: {len(gt_sl_pts)}")
        print(f"    {'Transform':<15}  {'Median err (m)':>15}  {'Mean err (m)':>13}")
        print(f"    {'─'*15}  {'─'*15}  {'─'*13}")

        transform_results = {}
        for tname, tfunc in TRANSFORMS.items():
            transformed = np.array([tfunc(xs, ys) for xs, ys in est_sl_pts])
            errs = np.linalg.norm(transformed - gt_sl_arr, axis=1)
            transform_results[tname] = {
                "median": float(np.median(errs)),
                "mean":   float(np.mean(errs)),
            }

        best_name = min(transform_results, key=lambda k: transform_results[k]["median"])

        for tname, res in sorted(transform_results.items(), key=lambda x: x[1]["median"]):
            marker = " ← BEST" if tname == best_name else ""
            print(f"    {tname:<15}  {res['median']:>15.2f}  {res['mean']:>13.2f}{marker}")

        best_err = transform_results[best_name]["median"]
        if best_name == "identity":
            verdict_str = "genuine calibration failure (no transform helps)"
        else:
            verdict_str = f"convention bug: '{best_name}' fixes it (residual {best_err:.1f}m)"
        print(f"    VERDICT: {verdict_str}")

        frame_verdicts[fname] = best_name

        # ── top-down plot ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        fig.patch.set_facecolor("#111")

        title = (f"[{rank}] {fname}  |  pos_err median={img_record['median_pos_err_m']:.1f}m  "
                 f"|  best transform: {best_name} ({best_err:.1f}m residual)")
        fig.suptitle(title, color="white", fontsize=11)

        # Left: GT positions (via our current synloc_to_pnlcalib transform)
        ax = axes[0]
        draw_pitch(ax)
        if gt_pnl_pts:
            gxs, gys = zip(*gt_pnl_pts)
            ax.scatter(gxs, gys, s=35, c="#4fc3f7", edgecolors="white",
                       linewidths=0.5, zorder=5, label="GT players")
        ax.set_title("GT players\n(SynLoc annotation → PnLCalib frame via current transform)",
                     color="white", fontsize=9)
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white"); ax.spines[:].set_color("#555")
        ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

        # Right: Estimated positions from H_inv (already in PnLCalib frame)
        ax = axes[1]
        draw_pitch(ax)
        if est_pnl_pts:
            exs, eys = zip(*est_pnl_pts)
            # Cap display range so extreme outliers don't collapse the good points
            in_range = [(x, y) for x, y in est_pnl_pts
                        if abs(x) < PNL_HALF_X * 5 and abs(y) < PNL_HALF_Y * 5]
            out_range = [(x, y) for x, y in est_pnl_pts
                         if abs(x) >= PNL_HALF_X * 5 or abs(y) >= PNL_HALF_Y * 5]
            if in_range:
                irx, iry = zip(*in_range)
                ax.scatter(irx, iry, s=35, c="#ff7043", edgecolors="white",
                           linewidths=0.5, zorder=5, label=f"Estimated ({len(in_range)})")
            if out_range:
                ax.text(0, 0, f"{len(out_range)} pts off-chart", color="yellow",
                        ha="center", va="center", fontsize=9, zorder=6)
        ax.set_title("Estimated players\n(GT pixel → H_inv → PnLCalib frame)",
                     color="white", fontsize=9)
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white"); ax.spines[:].set_color("#555")
        ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

        plot_path = os.path.join(out_dir, f"diag_worst_{rank}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    Plot saved → {plot_path}")

    # ── Overall verdict ───────────────────────────────────────────────────────
    print()
    print("  ── Check 1 Overall Verdict ─────────────────────────────────────────")
    if not frame_verdicts:
        print("  No frames could be classified.")
        return frame_verdicts

    counts = Counter(frame_verdicts.values())
    total  = len(frame_verdicts)
    print(f"  Transform classification across {total} analysed frames:")
    for tname, cnt in counts.most_common():
        print(f"    {tname:<15}: {cnt}/{total} frames")

    dominant_name, dominant_cnt = counts.most_common(1)[0]
    if dominant_name == "identity" and dominant_cnt >= total * 0.5:
        print("\n  VERDICT: GENUINE CALIBRATION FAILURES")
        print("           The 'identity' transform (i.e. no correction) is best for most")
        print("           bad frames.  No single axis flip/swap explains the large errors.")
        print("           This is per-frame calibration failure, not a convention bug.")
    elif dominant_name != "identity" and dominant_cnt >= total * 0.5:
        print(f"\n  VERDICT: SYSTEMATIC CONVENTION BUG — '{dominant_name}'")
        print(f"           {dominant_cnt}/{total} bad frames are best explained by this transform.")
        print(f"           This is fixable in one line of coordinate-conversion code.")
    else:
        print("\n  VERDICT: MIXED — no single transform dominates.")
        print("           Bad frames have different failure modes: some may be convention")
        print("           bugs, others genuine calibration failures or degenerate scenes.")

    return frame_verdicts


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Radial distance vs position error
# ════════════════════════════════════════════════════════════════════════════════

def check2(data, images_by_id, anns_by_img, out_dir):
    print()
    print("=" * 72)
    print("  CHECK 2 — Radial distance vs position error (fisheye test)")
    print("=" * 72)
    print("  (no GPU re-run; uses stored per-player errors + GT pixels from annotations)")

    calibrated = [r for r in data["per_image"] if r["status"] == "calibrated"]

    all_radial:   list[float] = []
    all_pos_err:  list[float] = []
    n_matched = 0

    for img_record in calibrated:
        image_id = img_record["image_id"]
        img_info = images_by_id.get(image_id)
        if img_info is None:
            continue

        cx = int(img_info["width"])  / 2.0
        cy = int(img_info["height"]) / 2.0

        anns            = anns_by_img[image_id]
        stored_errs     = img_record.get("player_pos_errs_m", [])

        j = 0
        for a in anns:
            foot_kp = a["keypoints"][1]
            if int(foot_kp[2]) == 0:
                continue
            u_gt = float(foot_kp[0])
            v_gt = float(foot_kp[1])
            radial = float(np.sqrt((u_gt - cx) ** 2 + (v_gt - cy) ** 2))

            if j < len(stored_errs):
                err = stored_errs[j]
                if err is not None and np.isfinite(err):
                    all_radial.append(radial)
                    all_pos_err.append(err)
                    n_matched += 1
            j += 1

    radial_arr  = np.array(all_radial)
    pos_err_arr = np.array(all_pos_err)

    print(f"\n  Matched {n_matched} player-points across {len(calibrated)} calibrated images")
    print(f"  Radial range : {radial_arr.min():.0f} – {radial_arr.max():.0f} px")
    print(f"  Error range  : {pos_err_arr.min():.2f} – {pos_err_arr.max():.2f} m")

    # Correlation (all points, and p99-clipped to damp outliers)
    corr_all = float(np.corrcoef(radial_arr, pos_err_arr)[0, 1])
    p99      = float(np.percentile(pos_err_arr, 99))
    mask     = pos_err_arr <= p99
    corr_p99 = float(np.corrcoef(radial_arr[mask], pos_err_arr[mask])[0, 1])

    print(f"\n  Pearson r (all points)    : {corr_all:+.4f}")
    print(f"  Pearson r (≤p99 err clip) : {corr_p99:+.4f}  [n={mask.sum()}]")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    sc = ax.scatter(
        radial_arr[mask], pos_err_arr[mask],
        alpha=0.20, s=6, c=pos_err_arr[mask], cmap="plasma",
        vmin=0, vmax=float(np.percentile(pos_err_arr[mask], 95)), rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Position error (m)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Binned median line
    n_bins = 16
    bin_edges = np.linspace(radial_arr[mask].min(), radial_arr[mask].max(), n_bins + 1)
    bx, by, bstd = [], [], []
    for i in range(n_bins):
        in_bin = (radial_arr[mask] >= bin_edges[i]) & (radial_arr[mask] < bin_edges[i + 1])
        if in_bin.sum() >= 5:
            bx.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            vals = pos_err_arr[mask][in_bin]
            by.append(float(np.median(vals)))
            bstd.append(float(np.percentile(vals, 75) - np.percentile(vals, 25)))

    if bx:
        bx, by, bstd = np.array(bx), np.array(by), np.array(bstd)
        ax.fill_between(bx, np.maximum(0, by - bstd / 2), by + bstd / 2,
                        alpha=0.25, color="cyan")
        ax.plot(bx, by, "c-", lw=2.5, label="bin median ± IQR/2", zorder=6)
        ax.plot(bx, by, "wo", ms=5, zorder=7)

    ax.set_xlabel("Radial distance from image centre (pixels)", fontsize=12, color="white")
    ax.set_ylabel("Position error (metres)", fontsize=12, color="white")
    ax.set_title(
        f"Radial pixel distance vs position error in metres\n"
        f"Pearson r = {corr_p99:+.3f}  (p99-clipped, n={mask.sum()})",
        color="white", fontsize=12,
    )
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")

    plot_path = os.path.join(out_dir, "diag_radial_vs_error.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Scatter plot saved → {plot_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    r = corr_p99
    if r >= 0.40:
        print(f"  VERDICT: STRONG POSITIVE correlation (r={r:+.3f})")
        print("           Position error grows toward frame edges.")
        print("           This is CONSISTENT with fisheye distortion: PnLCalib's")
        print("           pinhole model under-corrects pixels at large radii.")
        print("           Fisheye is likely the dominant source of the 15.8m median error.")
    elif r >= 0.20:
        print(f"  VERDICT: WEAK correlation (r={r:+.3f})")
        print("           Some radial trend but not clearly fisheye-dominated.")
        print("           Error sources are mixed: fisheye + scale/frame issues.")
    elif r <= -0.20:
        print(f"  VERDICT: NEGATIVE correlation (r={r:+.3f})  — unusual.")
        print("           Error is WORSE near image centre.  Not typical fisheye.")
    else:
        print(f"  VERDICT: NO significant radial correlation (r={r:+.3f})")
        print("           Error is approximately UNIFORM across the frame.")
        print("           Fisheye distortion is NOT the primary driver of the 15.8m error.")
        print("           Likely cause: scale mismatch or systematic frame/convention bug.")

    return corr_p99


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Diagnostic on existing synloc_calibration_eval.json run."
    )
    ap.add_argument("--eval-json", default=DEFAULT_EVAL_JSON,
                    help="Path to per-image eval JSON (default: synloc_calibration_eval.json)")
    ap.add_argument("--ann-file",  default=VAL_ANN_FILE,
                    help="SynLoc val annotation file")
    ap.add_argument("--val-dir",   default=VAL_IMG_DIR,
                    help="SynLoc val image directory")
    ap.add_argument("--out-dir",   default=DEFAULT_OUT_DIR,
                    help="Directory to write diagnostic PNG files")
    ap.add_argument("--n-worst",   type=int, default=6,
                    help="Number of worst frames to analyse in Check 1")
    args = ap.parse_args()

    # Resolve before chdir
    args.eval_json = os.path.abspath(args.eval_json)
    args.ann_file  = os.path.abspath(args.ann_file)
    args.val_dir   = os.path.abspath(args.val_dir)
    args.out_dir   = os.path.abspath(args.out_dir)

    print(f"Loading eval JSON  : {args.eval_json}")
    with open(args.eval_json) as f:
        data = json.load(f)

    print(f"Loading annotations: {args.ann_file}")
    with open(args.ann_file) as f:
        ann_data = json.load(f)

    images_by_id = {img["id"]: img for img in ann_data["images"]}
    anns_by_img  = defaultdict(list)
    for a in ann_data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_cal = sum(1 for r in data["per_image"] if r["status"] == "calibrated")
    print(f"Eval JSON: {len(data['per_image'])} images, {n_cal} calibrated")
    print(f"Annotations: {len(ann_data['images'])} images, {len(ann_data['annotations'])} players")
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nLoading PnLCalib models (needed for Check 1 H re-derivation)...")
    model_kp, model_l = load_models()
    print("Models loaded.\n")

    verdicts = check1(
        data, images_by_id, anns_by_img,
        model_kp, model_l, args.out_dir, args.n_worst,
    )

    corr = check2(data, images_by_id, anns_by_img, args.out_dir)

    print()
    print("=" * 72)
    print("  Diagnostic complete.")
    print(f"  Check 1: {len(verdicts)} worst frames classified, plots → {args.out_dir}/diag_worst_*.png")
    print(f"  Check 2: r={corr:+.3f}, scatter → {args.out_dir}/diag_radial_vs_error.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
