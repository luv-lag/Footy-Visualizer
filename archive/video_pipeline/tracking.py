"""Nearest-neighbour player tracking across keyframes."""

import numpy as np

MATCH_THRESHOLD = 250.0  # pixels
MIN_CONSECUTIVE_KEYFRAMES = 4


def centres(detections: list[dict]) -> np.ndarray:
    """Return (N, 2) array of [x, y] centres."""
    if not detections:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[d["x"], d["y"]] for d in detections], dtype=np.float32)


def match_detections(prev: np.ndarray, curr: np.ndarray) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour matching between two sets of centres.

    Returns list of (prev_idx, curr_idx) pairs within MATCH_THRESHOLD.
    """
    if len(prev) == 0 or len(curr) == 0:
        return []

    matched = []
    used_curr = set()
    for i, p in enumerate(prev):
        dists = np.linalg.norm(curr - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] < MATCH_THRESHOLD and j not in used_curr:
            matched.append((i, j))
            used_curr.add(j)
    return matched


def _min_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return the minimum centre distance between any player in a and any in b."""
    if len(a) == 0 or len(b) == 0:
        return None
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)  # (Na, Nb)
    return float(dists.min())


def build_tracks(all_detections: list[list[dict]]) -> list[list[dict | None]]:
    """Track players across keyframes using nearest-neighbour matching.

    Returns a list of tracks; each track is a list of length len(all_detections)
    with None where the player was not visible and a detection dict otherwise.
    Only tracks present in at least MIN_CONSECUTIVE_KEYFRAMES consecutive frames
    are retained.
    """
    n_frames = len(all_detections)
    if n_frames == 0:
        return []

    # Debug: detection counts per keyframe
    print("[tracking] Keyframe detection counts:")
    for fi, dets in enumerate(all_detections):
        print(f"  KF {fi:>3}: {len(dets):>2} players detected")

    # Debug: minimum inter-frame centre distance per consecutive pair
    print("[tracking] Min centre distance between consecutive keyframe pairs:")
    for fi in range(n_frames - 1):
        a = centres(all_detections[fi])
        b = centres(all_detections[fi + 1])
        min_d = _min_pairwise_distance(a, b)
        if min_d is None:
            print(f"  KF {fi}-{fi+1}: no detections in one or both frames")
        else:
            flag = "  <<< ABOVE THRESHOLD" if min_d > MATCH_THRESHOLD else ""
            print(f"  KF {fi}-{fi+1}: min dist = {min_d:6.1f}px  (threshold={MATCH_THRESHOLD:.0f}px){flag}")

    # active_tracks: list of (track_list, last_centre)
    # track_list index corresponds to keyframe index
    active_tracks: list[list] = []

    for fi, detections in enumerate(all_detections):
        curr = centres(detections)
        if fi == 0:
            for det in detections:
                track = [None] * n_frames
                track[0] = det
                active_tracks.append(track)
            continue

        prev_centres = np.array(
            [t[fi - 1]["x"] if t[fi - 1] is not None else
             next((t[k]["x"] for k in range(fi - 1, -1, -1) if t[k] is not None), -9999)
             for t in active_tracks], dtype=np.float32
        )
        prev_centres_y = np.array(
            [t[fi - 1]["y"] if t[fi - 1] is not None else
             next((t[k]["y"] for k in range(fi - 1, -1, -1) if t[k] is not None), -9999)
             for t in active_tracks], dtype=np.float32
        )
        prev_pts = np.stack([prev_centres, prev_centres_y], axis=1)

        matches = match_detections(prev_pts, curr)
        matched_curr = {j for _, j in matches}

        for ti, cj in matches:
            active_tracks[ti][fi] = detections[cj]

        # Start new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_curr:
                track = [None] * n_frames
                track[fi] = det
                active_tracks.append(track)

    # Filter: keep only tracks with >= MIN_CONSECUTIVE_KEYFRAMES consecutive hits
    stable = []
    for track in active_tracks:
        presence = [1 if t is not None else 0 for t in track]
        max_run = _max_consecutive(presence)
        if max_run >= MIN_CONSECUTIVE_KEYFRAMES:
            stable.append(track)

    print(f"[tracking] {len(stable)} stable tracks retained (≥{MIN_CONSECUTIVE_KEYFRAMES} consecutive keyframes)")
    return stable


def _max_consecutive(seq: list[int]) -> int:
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def stable_point_pairs(
    tracks: list[list],
    frame_a: int,
    frame_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched centre arrays for two keyframe indices from stable tracks."""
    pts_a, pts_b = [], []
    for track in tracks:
        da, db = track[frame_a], track[frame_b]
        if da is not None and db is not None:
            pts_a.append([da["x"], da["y"]])
            pts_b.append([db["x"], db["y"]])
    if not pts_a:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.array(pts_a, dtype=np.float32), np.array(pts_b, dtype=np.float32)
