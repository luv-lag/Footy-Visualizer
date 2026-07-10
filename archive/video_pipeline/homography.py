"""Homography computation, focal length estimation, decomposition, LM refinement."""

import numpy as np
import cv2

FOCAL_MIN = 500.0
FOCAL_MAX = 5000.0
STANDARD_PITCH_WIDTH_M = 68.0
REPROJECTION_OUTLIER_THRESHOLD = 100.0  # pixels


# ---------------------------------------------------------------------------
# Focal length estimation (Hartley-Zisserman)
# ---------------------------------------------------------------------------

def _h_to_focal_candidates(H: np.ndarray) -> list[float]:
    """Extract focal length candidates from a homography.

    Based on the constraint that for a pure-rotation homography
    K^{-1} H should have two equal singular values when K = diag(f, f, 1).
    We sweep over f and minimise |s1 - s2| where s1, s2 are the two
    largest singular values of K^{-1} H K.
    """
    # Closed-form via the two-focal-length formula from Hartley & Zisserman
    # (Algorithm from the Zhang/Sturm focal-from-homography derivation)
    h = H.ravel()
    h00, h01, h02 = h[0], h[1], h[2]
    h10, h11, h12 = h[3], h[4], h[5]
    h20, h21, h22 = h[6], h[7], h[8]

    # Use the two constraints derived from H^T K^{-T} K^{-1} H = I (rotation)
    # Constraint 1: h0 . h1 = 0  =>  h00*h10 + h01*h11 + h02*h12/f^2 = 0
    # Constraint 2: |h0|^2 = |h1|^2 => ... solve for f

    # Numerically solve using SVD sweep
    candidates = []
    for f in np.linspace(FOCAL_MIN, FOCAL_MAX, 1000):
        K_inv = np.array([[1 / f, 0, 0],
                          [0, 1 / f, 0],
                          [0, 0, 1]], dtype=np.float64)
        M = K_inv @ H
        _, sv, _ = np.linalg.svd(M)
        # For pure rotation the two larger SVs should be equal
        candidates.append((abs(sv[0] - sv[1]), f))

    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:3]]


def estimate_focal_length(H: np.ndarray, image_width: int) -> float:
    """Estimate focal length from homography. Falls back if outside valid range."""
    try:
        candidates = _h_to_focal_candidates(H)
        f = candidates[0]
        if FOCAL_MIN <= f <= FOCAL_MAX:
            return float(f)
    except Exception:
        pass
    return float(image_width) * 1.2


def build_K(f: float, w: int, h: int) -> np.ndarray:
    """Construct 3x3 camera intrinsic matrix."""
    return np.array([
        [f,   0,  w / 2.0],
        [0,   f,  h / 2.0],
        [0,   0,  1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def compute_homography(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray | None:
    """Compute homography from frame A to frame B with RANSAC."""
    if len(pts_a) < 4:
        return None
    H, mask = cv2.findHomography(
        pts_a.reshape(-1, 1, 2),
        pts_b.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return H


# ---------------------------------------------------------------------------
# Decomposition and ambiguity resolution
# ---------------------------------------------------------------------------

def _positive_depth_filter(
    R: np.ndarray,
    t: np.ndarray,
    pts2d_a: np.ndarray,
    pts2d_b: np.ndarray,
    K: np.ndarray,
) -> bool:
    """Return True if all triangulated points have positive depth in both views."""
    if len(pts2d_a) == 0:
        return False
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts2d_a.T.reshape(2, -1).astype(np.float32),
        pts2d_b.T.reshape(2, -1).astype(np.float32),
    )
    pts3d = pts4d[:3] / pts4d[3]
    # Check depth in camera 1
    if np.any(pts3d[2] <= 0):
        return False
    # Check depth in camera 2
    pts_cam2 = R @ pts3d + t.reshape(3, 1)
    if np.any(pts_cam2[2] <= 0):
        return False
    return True


def _reprojection_error(
    R: np.ndarray,
    t: np.ndarray,
    pts2d_a: np.ndarray,
    pts2d_b: np.ndarray,
    K: np.ndarray,
) -> float:
    """Mean reprojection error of pts_b given R, t from pts_a (via triangulation)."""
    if len(pts2d_a) == 0:
        return float("inf")
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts2d_a.T.reshape(2, -1).astype(np.float32),
        pts2d_b.T.reshape(2, -1).astype(np.float32),
    )
    pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)
    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float32),
        cv2.Rodrigues(R)[0],
        t.astype(np.float32),
        K.astype(np.float32),
        None,
    )
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - pts2d_b, axis=1)))


def decompose_homography(
    H: np.ndarray,
    K: np.ndarray,
    pts2d_a: np.ndarray,
    pts2d_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Decompose H, resolve 4-way ambiguity, return (R, t, mean_reproj_error)."""
    num_solutions, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    valid = []
    for i in range(num_solutions):
        R, t = Rs[i], ts[i]
        if _positive_depth_filter(R, t, pts2d_a, pts2d_b, K):
            err = _reprojection_error(R, t, pts2d_a, pts2d_b, K)
            valid.append((R, t, err))

    if not valid:
        # Fall back: take all solutions and pick lowest error
        for i in range(num_solutions):
            R, t = Rs[i], ts[i]
            err = _reprojection_error(R, t, pts2d_a, pts2d_b, K)
            valid.append((R, t, err))

    if not valid:
        return None

    valid.sort(key=lambda x: x[2])
    return valid[0]  # (R, t, error)


# ---------------------------------------------------------------------------
# LM refinement
# ---------------------------------------------------------------------------

def _triangulate_pts3d(
    R: np.ndarray,
    t: np.ndarray,
    pts2d_a: np.ndarray,
    pts2d_b: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate and return (pts3d, inlier_mask) excluding high-error points."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts2d_a.T.reshape(2, -1).astype(np.float32),
        pts2d_b.T.reshape(2, -1).astype(np.float32),
    )
    pts3d = (pts4d[:3] / pts4d[3]).T

    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float32), rvec, t.astype(np.float32), K.astype(np.float32), None
    )
    proj = proj.reshape(-1, 2)
    errs = np.linalg.norm(proj - pts2d_b, axis=1)
    inlier_mask = errs < REPROJECTION_OUTLIER_THRESHOLD
    return pts3d, inlier_mask


def refine_pose(
    R: np.ndarray,
    t: np.ndarray,
    pts2d_b: np.ndarray,
    pts3d: np.ndarray,
    K: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine R, t with Levenberg-Marquardt using inlier correspondences."""
    pts3d_in = pts3d[inlier_mask].astype(np.float32)
    pts2d_in = pts2d_b[inlier_mask].astype(np.float32)
    if len(pts3d_in) < 4:
        return R, t
    rvec_init, _ = cv2.Rodrigues(R)
    try:
        rvec_ref, t_ref = cv2.solvePnPRefineLM(
            pts3d_in.reshape(-1, 1, 3),
            pts2d_in.reshape(-1, 1, 2),
            K.astype(np.float32),
            None,
            rvec_init.astype(np.float32),
            t.reshape(3, 1).astype(np.float32),
        )
        R_ref, _ = cv2.Rodrigues(rvec_ref)
        return R_ref, t_ref.ravel()
    except cv2.error:
        return R, t


# ---------------------------------------------------------------------------
# Scale recovery and camera parameter extraction
# ---------------------------------------------------------------------------

def recover_camera_params(
    R: np.ndarray,
    t: np.ndarray,
    pts2d_a: np.ndarray,
    pitch_width_px: float,
    f: float,
) -> tuple[float, float]:
    """Return (distance_m, elevation_deg) from recovered pose and scale.

    pitch_width_px: visible horizontal pitch width in pixels.
    """
    scale = STANDARD_PITCH_WIDTH_M / pitch_width_px if pitch_width_px > 0 else 1.0
    t_m = t * scale
    d = float(np.linalg.norm(t_m))

    # Elevation: angle between camera Y axis and pitch plane (XZ plane)
    # R maps world axes to camera frame; pitch normal is world Y
    # camera up in world = R.T @ [0, 1, 0]
    cam_up_world = R.T @ np.array([0.0, 1.0, 0.0])
    # elevation is angle of translation vector below horizon
    if d > 0:
        t_norm = t_m / d
        # elevation angle relative to ground plane
        theta = float(np.degrees(np.arcsin(np.clip(-t_norm[1], -1, 1))))
    else:
        theta = 0.0

    return d, theta


def estimate_pitch_width_px(pts2d_a: np.ndarray) -> float:
    """Estimate visible pitch width in pixels from leftmost to rightmost player."""
    if len(pts2d_a) == 0:
        return 1.0
    return float(pts2d_a[:, 0].max() - pts2d_a[:, 0].min())
