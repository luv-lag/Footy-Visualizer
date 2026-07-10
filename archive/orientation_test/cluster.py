import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_


def pca_cluster(points: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, sv, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    aspect_ratio = sv[0] / sv[1] if sv[1] > 1e-10 else float("inf")
    return direction, angle, aspect_ratio, centroid


def analyze_clusters(
    labels: np.ndarray, coords: np.ndarray, min_aspect_ratio: float = 2.5
) -> list[dict]:
    clusters = []
    for lbl in sorted(set(labels) - {-1}):
        pts = coords[labels == lbl]
        if len(pts) < 2:
            continue
        direction, angle, ar, centroid = pca_cluster(pts)
        clusters.append(
            {
                "label": int(lbl),
                "points": pts,
                "centroid": centroid,
                "direction": direction,
                "angle": angle,
                "aspect_ratio": ar,
                "count": len(pts),
                "passes": ar >= min_aspect_ratio,
            }
        )
    return clusters


def weighted_average_angle(clusters: list[dict]) -> float:
    passing = [c for c in clusters if c["passes"]]
    rads = np.array([np.radians(c["angle"]) for c in passing])
    w = np.array([c["count"] for c in passing], dtype=float)
    return np.degrees(np.arctan2(np.dot(w, np.sin(rads)), np.dot(w, np.cos(rads))))


def rotate_coords(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.radians(-angle_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return (R @ coords.T).T
