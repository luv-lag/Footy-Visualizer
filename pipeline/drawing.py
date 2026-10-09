"""drawing.py -- shared drawing conventions for the pipeline's renderers.

Class/team colors (OpenCV BGR) and resolution-relative sizing helpers, used
by ``tactical_map`` (the live app) and ``eval/debug_tools`` (diagnostics).
They live here, inside ``pipeline/``, so production code never has to
import from ``eval/``.
"""

from __future__ import annotations

import numpy as np

from detection import Detection

#: Fixed, visually distinct colors per class, in OpenCV's BGR order.
#: Chosen to stay far from pitch green and from each other.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (245, 135, 66),  # azure blue
    "goalkeeper": (0, 140, 255),  # orange
    "referee": (0, 0, 230),  # red
    "ball": (0, 230, 255),  # yellow
}

#: Colors for team-based coloring, keyed by team_id. Team 0 reuses the player
#: class color (players are still "players" visually); team 1 gets a color
#: far from every other color in use (class colors, pitch green) so the two
#: teams are never ambiguous with each other or with goalkeeper/referee/ball.
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: CLASS_COLORS["player"],  # azure blue
    1: (203, 45, 224),  # magenta
}

#: Reference diagonal (720p) that scale-dependent drawing sizes are relative to.
_REFERENCE_DIAGONAL = float(np.hypot(1280, 720))


def scale_factor(image: np.ndarray) -> float:
    """Compute a resolution-relative scale factor for an image.

    Used so that line thickness, marker size, font size, and padding look
    proportionally correct on both a 720p frame and a 4K frame, instead of
    using fixed pixel values tuned for one resolution.

    Args:
        image: The image to scale against.

    Returns:
        A scale factor, 1.0 at 1280x720, scaling linearly with the image
        diagonal.
    """
    height, width = image.shape[:2]
    diagonal = float(np.hypot(width, height))
    return diagonal / _REFERENCE_DIAGONAL


def bbox_corners(detection: Detection) -> tuple[int, int, int, int]:
    """Convert a Detection's center/size representation to corner points.

    Args:
        detection: The detection to convert.

    Returns:
        (x1, y1, x2, y2) integer pixel coordinates of the bounding box.
    """
    cx, cy = detection.bbox_center
    w, h = detection.bbox_size
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return x1, y1, x2, y2
