import numpy as np
from sklearn.cluster import DBSCAN


def recover_formation(
    rotated_coords: np.ndarray, eps: float
) -> tuple[str, list[tuple[float, int]]]:
    y = rotated_coords[:, 1].reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=1).fit(y).labels_

    line_groups: dict[int, list[float]] = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        line_groups.setdefault(lbl, []).append(rotated_coords[i, 1])

    lines = sorted(
        [(float(np.mean(ys)), len(ys)) for ys in line_groups.values()],
        key=lambda x: x[0],
    )

    counts = [int(c) for _, c in lines]
    return "-".join(str(c) for c in counts), lines
