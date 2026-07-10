"""Player-formation line extraction (optional solver input).

A settled defensive line (back four/five) stands along a genuinely straight
real-world line, almost always parallel to its own goal line.  Fitting a
line through same-team player ground points therefore yields an image line
that passes through the *transverse* vanishing point -- a "found"
correspondence sourced from formation rather than paint, valuable exactly
when markings are sparse (e.g. a tight midfield shot showing only the
halfway line and centre circle).

This is a probabilistic signal: teams in transition / broken lines violate
the assumption.  The solver only uses it when paint-based strategies fail,
and flags + down-weights the result (see ``used_players`` on candidates and
``used_player_formation`` in calibration diagnostics).
"""

from __future__ import annotations

import numpy as np

from .geometry import normalize_line


def formation_lines(
    feet_xy: np.ndarray,
    labels: np.ndarray | None,
    image_size: tuple[int, int],
    min_players: int = 4,
    max_lines: int = 3,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """RANSAC candidate formation lines from player ground points.

    Args:
        feet_xy: (N, 2) player ground-contact points in *distorted* pixel
            coords (the caller undistorts before use if needed).
        labels: optional (N,) team labels; lines are fit within one team
            (a defensive line is single-team).  ``None`` fits across all.
        image_size: (w, h), used to scale tolerances.
        min_players: minimum collinear players to accept a line.
        max_lines: maximum number of lines returned (best first).

    Returns:
        Homogeneous image lines, best-supported first.  May be empty.
    """
    rng = rng or np.random.default_rng(23)
    feet_xy = np.asarray(feet_xy, dtype=float)
    if len(feet_xy) < min_players:
        return []
    w, h = image_size
    diag = float(np.hypot(w, h))
    tol = 0.012 * diag
    min_extent = 0.15 * diag

    groups: list[np.ndarray]
    if labels is not None:
        labels = np.asarray(labels)
        groups = [feet_xy[labels == v] for v in np.unique(labels)]
    else:
        groups = [feet_xy]

    scored: list[tuple[float, np.ndarray]] = []
    for pts in groups:
        if len(pts) < min_players:
            continue
        used = np.zeros(len(pts), bool)
        for _ in range(2):  # up to two lines per team (defensive + midfield)
            avail = np.nonzero(~used)[0]
            if len(avail) < min_players:
                break
            best_inl: np.ndarray | None = None
            for _ in range(200):
                i, j = rng.choice(avail, 2, replace=False)
                d = pts[j] - pts[i]
                nd = np.linalg.norm(d)
                if nd < min_extent * 0.5:
                    continue
                nrm = np.array([-d[1], d[0]]) / nd
                dist = np.abs((pts - pts[i]) @ nrm)
                inl = (dist < tol) & ~used
                if best_inl is None or inl.sum() > best_inl.sum():
                    best_inl = inl
            if best_inl is None or best_inl.sum() < min_players:
                break
            sel = pts[best_inl]
            extent = float(np.linalg.norm(sel.max(axis=0) - sel.min(axis=0)))
            if extent < min_extent:
                break
            # Total least squares through the inliers.
            c = sel.mean(axis=0)
            q = sel - c
            _, _, vt = np.linalg.svd(q, full_matrices=False)
            n = vt[-1]
            line = normalize_line(np.array([n[0], n[1], -n @ c]))
            scored.append((float(best_inl.sum()) + extent / diag, line))
            used |= best_inl
    scored.sort(key=lambda t: -t[0])
    return [l for _, l in scored[:max_lines]]
