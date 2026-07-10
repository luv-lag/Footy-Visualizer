"""
calibration.py — PnLCalib wrapper for the production pipeline.

World coordinate frame (PnLCalib / FIFA standard):
  Origin : pitch centre
  X axis : length (long axis), half-extent ±52.5 m   (total 105 m)
  Y axis : width  (short axis), half-extent ±34.0 m  (total  68 m)
  Z axis : up (positive above ground plane)
  H maps ground-plane world [Xw, Yw, 1] → image pixel [u/w, v/w].
  H_inv  maps image pixel   [u,  v,  1] → ground-plane world (up to scale).

All constants and the calibrate() logic are lifted verbatim from the
original evaluate_synloc_calibration.py (calibrate_image() function, now
archived under eval/synloc/) and vendor/PnLCalib/extract_transform.py.
Numerical behaviour is identical to those scripts.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
from PIL import Image

# ── PnLCalib path resolution ──────────────────────────────────────────────────
# calibration.py lives at: pipeline/pnlcalib_wrapper/calibration.py
# PnLCalib lives at:       vendor/PnLCalib/
# NOTE: this directory depth (2 levels up to project root) is load-bearing --
# see the project README's path-fragility note before moving this file again.
_HERE        = os.path.dirname(os.path.abspath(__file__))   # .../pipeline/pnlcalib_wrapper
_ROOT        = os.path.dirname(os.path.dirname(_HERE))       # project root
PNLCALIB_DIR = os.path.join(_ROOT, "vendor", "PnLCalib")
assert os.path.isdir(PNLCALIB_DIR), (
    f"PNLCALIB_DIR does not exist: {PNLCALIB_DIR!r} -- pnlcalib_wrapper's "
    "directory depth assumption (2 levels below project root) no longer "
    "holds; fix the _ROOT computation above before anything else."
)

if PNLCALIB_DIR not in sys.path:
    sys.path.insert(0, PNLCALIB_DIR)

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

# ── Constants (lifted verbatim from evaluate_synloc_calibration.py) ───────────
DEVICE                 = "cuda:0"
KP_THRESHOLD           = 0.3434
LINE_THRESHOLD         = 0.7867
MIN_KP_FOR_VALID_CALIB = 4


# ── Result types ──────────────────────────────────────────────────────────────

class CalibGrade(str, Enum):
    PASS    = "pass"     # H is valid; proceed to projection
    ABSTAIN = "abstain"  # calibration failed; do not project players


@dataclass
class CalibResult:
    """Full camera calibration result.

    If grade == ABSTAIN, H / K / R / t / cam_pos are None.
    All non-None matrices use the PnLCalib world frame documented above.
    """
    grade      : CalibGrade
    H          : np.ndarray | None   # (3,3) ground-plane homography
    K          : np.ndarray | None   # (3,3) intrinsic matrix
    R          : np.ndarray | None   # (3,3) rotation (world → camera)
    t          : np.ndarray | None   # (3,) translation in camera frame
    cam_pos    : np.ndarray | None   # (3,) camera centre in world metres
    n_kp       : int                 # keypoints detected above threshold
    reproj_err : float               # PnLCalib reprojection error (px)
    mode       : str                 # heuristic_voting branch used
    pan_deg    : float | None        # camera pan  (degrees); None if ABSTAIN
    tilt_deg   : float | None        # camera tilt (degrees); None if ABSTAIN
    roll_deg   : float | None        # camera roll (degrees); None if ABSTAIN


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device: str = DEVICE) -> tuple:
    """Load SV_kp + SV_lines HRNet-W48 models from PnLCalib weights.

    Logic lifted from evaluate_synloc_calibration.load_models().
    PnLCalib opens config/*.yaml and weights/* with CWD-relative paths, so we
    chdir into PNLCALIB_DIR for the duration of the load and restore afterwards.
    """
    orig = os.getcwd()
    os.chdir(PNLCALIB_DIR)
    try:
        cfg   = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))

        torch.cuda.init()

        model_kp = get_cls_net(cfg)
        model_kp.load_state_dict(torch.load("weights/SV_kp", map_location=device))
        model_kp.to(device).eval()

        model_l = get_cls_net_l(cfg_l)
        model_l.load_state_dict(torch.load("weights/SV_lines", map_location=device))
        model_l.to(device).eval()
    finally:
        os.chdir(orig)

    return model_kp, model_l


# ── Per-frame calibration ─────────────────────────────────────────────────────

def calibrate(
    frame_bgr : np.ndarray,
    model_kp,
    model_l,
    device    : str = DEVICE,
) -> CalibResult:
    """Run PnLCalib on one frame and return a CalibResult.

    Core logic lifted verbatim from evaluate_synloc_calibration.calibrate_image()
    and PnLCalib/extract_transform.py. Numerical behaviour is identical.

    The verification gate (result is None OR n_kp < MIN_KP_FOR_VALID_CALIB) from
    the inline logic of calibrate_image() is made explicit here as CalibGrade.
    """
    h_orig, w_orig = frame_bgr.shape[:2]

    # Preprocess — lifted from calibrate_image() verbatim
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = tvf.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    if img_t.size()[-1] != 960:
        img_t = T.Resize((540, 960))(img_t)
    img_t = img_t.to(device)
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

    # ── Verification gate (lifted from calibrate_image() inline logic) ────────
    if result is None:
        return CalibResult(
            grade=CalibGrade.ABSTAIN, H=None, K=None, R=None, t=None,
            cam_pos=None, n_kp=n_kp, reproj_err=float("inf"), mode="fail",
            pan_deg=None, tilt_deg=None, roll_deg=None,
        )

    if n_kp < MIN_KP_FOR_VALID_CALIB:
        return CalibResult(
            grade=CalibGrade.ABSTAIN, H=None, K=None, R=None, t=None,
            cam_pos=None, n_kp=n_kp, reproj_err=float(result["rep_err"]),
            mode="degenerate",
            pan_deg=None, tilt_deg=None, roll_deg=None,
        )

    # ── K, R, t, H assembly (lifted from extract_transform.py lines 103–116) ──
    cp  = result["cam_params"]
    K   = np.array([
        [cp["x_focal_length"], 0.0, cp["principal_point"][0]],
        [0.0, cp["y_focal_length"], cp["principal_point"][1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    R   = np.array(cp["rotation_matrix"], dtype=np.float64)
    pos = np.array(cp["position_meters"],  dtype=np.float64)
    t   = -(R @ pos)

    H   = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H  /= H[2, 2]

    return CalibResult(
        grade=CalibGrade.PASS,
        H=H, K=K, R=R, t=t, cam_pos=pos,
        n_kp=n_kp,
        reproj_err=float(result["rep_err"]),
        mode=str(result["mode"]),
        pan_deg=float(cp["pan_degrees"]),
        tilt_deg=float(cp["tilt_degrees"]),
        roll_deg=float(cp["roll_degrees"]),
    )
