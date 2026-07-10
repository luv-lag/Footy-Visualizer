"""DBSCAN formation detection on warped-frame player positions."""

import numpy as np
from sklearn.cluster import DBSCAN

# Heuristic: two teams separated by X in the warped (top-down) view
TEAM_X_SPLIT_FRACTION = 0.5

# DBSCAN params for line grouping on Y coordinate (depth)
DBSCAN_EPS = 40        # pixels in warped frame
DBSCAN_MIN_SAMPLES = 1


def assign_teams(
    player_pts: np.ndarray, frame_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split players into two teams by median X in the warped frame."""
    if len(player_pts) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    mid = frame_w * TEAM_X_SPLIT_FRACTION
    team_a = np.where(player_pts[:, 0] < mid)[0]
    team_b = np.where(player_pts[:, 0] >= mid)[0]
    return team_a, team_b


def formation_string(y_coords: np.ndarray) -> str:
    """Cluster Y positions and return a formation string like '4-3-3'."""
    if len(y_coords) == 0:
        return ""
    y = y_coords.reshape(-1, 1)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(y)
    unique = [l for l in np.unique(labels) if l >= 0]
    if not unique:
        return str(len(y_coords))
    line_counts = sorted(
        [int(np.sum(labels == l)) for l in unique],
        reverse=True,
    )
    return "-".join(str(c) for c in line_counts)


def detect_formations(
    warped_detections: list[dict],
    warp_matrix: np.ndarray,
    inv_warp: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> dict:
    """Run formation detection and return result dict.

    warped_detections: list of dicts with 'x', 'y' in *original* frame coords.
    warp_matrix: 3x3 perspective transform to top-down.
    inv_warp: inverse of warp_matrix.
    """
    if not warped_detections:
        return {
            "team_a_formation": "",
            "team_b_formation": "",
            "team_a_players": [],
            "team_b_players": [],
        }

    orig_pts = np.array([[d["x"], d["y"]] for d in warped_detections], dtype=np.float32)

    # Warp to top-down
    pts_h = np.concatenate([orig_pts, np.ones((len(orig_pts), 1))], axis=1).T
    warped_h = warp_matrix @ pts_h
    warped_pts = (warped_h[:2] / warped_h[2]).T  # (N, 2)

    team_a_idx, team_b_idx = assign_teams(warped_pts, frame_w)

    def team_info(idx: np.ndarray, warped: np.ndarray, orig: np.ndarray) -> tuple:
        if len(idx) == 0:
            return "", []
        y_coords = warped[idx, 1]
        fs = formation_string(y_coords)
        # Inverse-transform back to original frame
        team_warped = warped[idx]
        t_h = np.concatenate([team_warped, np.ones((len(team_warped), 1))], axis=1).T
        inv_h = inv_warp @ t_h
        inv_pts = (inv_h[:2] / inv_h[2]).T
        classes = [warped_detections[i].get("class", "player") for i in idx]
        return fs, [
            {"x": float(inv_pts[j, 0]), "y": float(inv_pts[j, 1]),
             "class": classes[j]}
            for j in range(len(idx))
        ]

    fa, pa = team_info(team_a_idx, warped_pts, orig_pts)
    fb, pb = team_info(team_b_idx, warped_pts, orig_pts)

    return {
        "team_a_formation": fa,
        "team_b_formation": fb,
        "team_a_players": pa,
        "team_b_players": pb,
    }
