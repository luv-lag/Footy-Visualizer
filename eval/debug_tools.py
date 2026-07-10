"""Visual debugging tools for the detection stage of the football pipeline.

detection.py answers "what is in this frame and where, in pixel coordinates."
Before that data is trusted by downstream stages (team assignment,
homography, drawing), these tools let you *look* at the raw output and catch
problems -- e.g. a foot point drifting off a player's feet, a misclassified
referee, a bounding box a half-frame too large -- at a glance.

Everything here is diagnostic only: no clustering, no homography, no pitch
overlay. Four functions are provided:

    debug_foot_points(image_path, detections)
        Sanity-check foot-point extraction in isolation.
    debug_raw_detections(image_path, detections)
        Sanity-check the full raw detection set across all four classes.
    debug_annotated_frame(image_path, detections, team_result=None)
        The "final" pixel-space annotated frame: bbox + class/team + confidence.
        Colored by class by default; pass the dict returned by
        team_separation.assign_teams() as team_result to color players by
        team instead (see that function's docstring for details, including
        how a low-confidence team split is flagged directly on the image).
    debug_combined(image_path, detections, team_result=None)
        All three views above, stacked into a single labeled image -- useful
        when reviewing many frames so each frame's diagnostics live in one
        file instead of three.

Each function saves a timestamped JPEG under ./debug_output/ and prints the
saved path (the functions return None; the printed path is the interface for
locating output, including from the __main__ block below).

Note: this module has no import-time dependency on team_separation.py --
debug_annotated_frame/debug_combined just accept team_result as a plain
dict matching assign_teams()'s return shape, so debug_tools.py stays usable
on its own. The __main__ block below imports team_separation lazily, only
when the script is actually run, to demonstrate the two modules together.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../eval
_ROOT = os.path.dirname(_HERE)                        # project root
_PIPELINE_DIR = os.path.join(_ROOT, "pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from detection import Detection, filter_players, infer

#: Output directory for all debug images.
OUTPUT_DIR = os.path.join(_ROOT, ".cache", "debug_output")

#: Fixed, visually distinct colors per class, in OpenCV's BGR order.
#: Chosen to stay far from pitch green and from each other.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (245, 135, 66),  # azure blue
    "goalkeeper": (0, 140, 255),  # orange
    "referee": (0, 0, 230),  # red
    "ball": (0, 230, 255),  # yellow
}

#: Foot points are always a player concept -- reuse the player color so the
#: two tools stay visually consistent with each other.
FOOT_POINT_COLOR = CLASS_COLORS["player"]

#: Colors for team-based coloring in debug_annotated_frame, keyed by team_id
#: as produced by team_separation.assign_teams(). Team 0 reuses the player
#: class color (players are still "players" visually); team 1 gets a color
#: far from every other color in use (class colors, pitch green) so the two
#: teams are never ambiguous with each other or with goalkeeper/referee/ball.
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: CLASS_COLORS["player"],  # azure blue
    1: (203, 45, 224),  # magenta
}

#: Text / background colors used for labels, legend, and summary boxes.
TEXT_COLOR = (255, 255, 255)  # white
LABEL_BG_COLOR = (0, 0, 0)  # black

#: Full-width warning banner shown on debug_annotated_frame when the supplied
#: team_result flags low_confidence_frame -- deliberately loud and solid
#: (not semi-transparent) since an unreliable team split is the single most
#: important thing to notice when looking at that image.
WARNING_BANNER_BG = (0, 0, 200)  # red
WARNING_BANNER_TEXT = (255, 255, 255)  # white

#: Opacity for fixed HUD panels (legend, summary counts, foot-point count).
#: High enough that an accidental overlap with a detection underneath reads
#: as a clean occlusion rather than a confusing translucent ghost.
LABEL_BG_ALPHA = 0.8

#: Per-detection label backgrounds are fully opaque and use the exact class
#: color, so they always match the legend swatch -- a semi-transparent
#: background here would blend with the grass and no longer match.
DETECTION_LABEL_ALPHA = 1.0

#: Reference diagonal (720p) that scale-dependent drawing sizes are relative to.
_REFERENCE_DIAGONAL = float(np.hypot(1280, 720))


def _load_image(image_path: str) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a BGR numpy array.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If the file exists but could not be decoded as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not decode image (unsupported or corrupt file): {image_path}")
    return image


def _scale_factor(image: np.ndarray) -> float:
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


def _timestamp() -> str:
    """Return a filesystem-safe timestamp string for output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_output_dir() -> str:
    """Create OUTPUT_DIR if needed and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _bbox_corners(detection: Detection) -> tuple[int, int, int, int]:
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


def _draw_label(
    image: np.ndarray,
    text: str,
    top_left: tuple[int, int],
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int] = TEXT_COLOR,
    bg_color: tuple[int, int, int] = LABEL_BG_COLOR,
    alpha: float = LABEL_BG_ALPHA,
    padding: int = 4,
) -> tuple[int, int]:
    """Draw a single line of text on a semi-transparent background box.

    Guarantees legibility regardless of what's behind the text, which is
    required since these images are broadcast frames with busy backgrounds
    (crowd, grass, other players).

    Args:
        image: Image to draw on, modified in place.
        text: Text to render.
        top_left: (x, y) pixel coordinates of the box's top-left corner.
        font_scale: OpenCV font scale.
        thickness: Text stroke thickness.
        text_color: BGR color of the text itself.
        bg_color: BGR color of the background box.
        alpha: Opacity of the background box (0 = invisible, 1 = opaque).
        padding: Padding in pixels around the text inside the box.

    Returns:
        (box_width, box_height) of the drawn box, so callers can stack
        multiple labels vertically.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_width = text_width + 2 * padding
    box_height = text_height + baseline + 2 * padding

    x, y = top_left
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), bg_color, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    text_origin = (x + padding, y + padding + text_height)
    cv2.putText(
        image, text, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA
    )
    return box_width, box_height


def _draw_legend(
    image: np.ndarray,
    entries: list[tuple[str, tuple[int, int, int]]],
    scale: float,
    margin: int,
    top_offset: int = 0,
) -> None:
    """Draw a fixed top-left legend mapping label to color.

    Generic over (label, color) pairs so the same function serves both
    class-colored views (label = class name) and team-colored views
    (label = "team 0" / "team 1" / etc), without the legend needing to know
    which mode produced the labels.

    Args:
        image: Image to draw on, modified in place.
        entries: (label, color) pairs to display, in display order. Only
            labels that actually appear in the current detections should be
            passed, so the legend never implies something is present when
            it isn't.
        scale: Resolution-relative scale factor.
        margin: Margin in pixels from the image edges.
        top_offset: Additional vertical offset from the top, e.g. to sit
            below a warning banner. 0 means flush with `margin` as before.
    """
    if not entries:
        return

    font_scale = max(0.4, 0.5 * scale)
    thickness = max(1, round(1.5 * scale))
    swatch_size = max(10, round(16 * scale))
    padding = max(4, round(6 * scale))
    line_gap = max(2, round(4 * scale))

    x, y = margin, margin + top_offset
    for label, color in entries:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        row_height = max(swatch_size, text_height + baseline) + 2 * padding
        row_width = swatch_size + padding + text_width + 2 * padding

        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + row_width, y + row_height), LABEL_BG_COLOR, -1)
        cv2.addWeighted(overlay, LABEL_BG_ALPHA, image, 1 - LABEL_BG_ALPHA, 0, dst=image)

        swatch_top = y + (row_height - swatch_size) // 2
        cv2.rectangle(
            image,
            (x + padding, swatch_top),
            (x + padding + swatch_size, swatch_top + swatch_size),
            color,
            -1,
            cv2.LINE_AA,
        )

        text_origin = (x + padding + swatch_size + padding, y + padding + text_height)
        cv2.putText(image, label, text_origin, font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

        y += row_height + line_gap


def _draw_summary_counts(
    image: np.ndarray, counts: Counter, scale: float, margin: int, top_offset: int = 0
) -> None:
    """Draw a per-label detection count summary in the top-right corner.

    Positioned opposite the legend (top-left) so the two never overlap.

    Args:
        image: Image to draw on, modified in place.
        counts: Mapping of display label (class name, or "team 0" / "team 1"
            in team-colored views) -> number of detections.
        scale: Resolution-relative scale factor.
        margin: Margin in pixels from the image edges.
        top_offset: Additional vertical offset from the top, e.g. to sit
            below a warning banner. 0 means flush with `margin` as before.
    """
    font_scale = max(0.4, 0.5 * scale)
    thickness = max(1, round(1.5 * scale))
    line_gap = max(2, round(4 * scale))
    height, width = image.shape[:2]

    lines = [f"{name}: {count}" for name, count in sorted(counts.items())]
    if not lines:
        lines = ["no detections"]

    y = margin + top_offset
    for line in lines:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        box_width = text_width + 8
        box_height = text_height + baseline + 8
        x = width - margin - box_width
        _draw_label(image, line, (x, y), font_scale, thickness, padding=4)
        y += box_height + line_gap


def _render_detections(
    image: np.ndarray,
    detections: list[Detection],
    scale: float,
    show_confidence: bool,
    team_assignments: dict[int, dict] | None = None,
) -> tuple[Counter, dict[str, tuple[int, int, int]]]:
    """Draw bounding boxes and labels for every detection.

    Shared by debug_raw_detections and debug_annotated_frame. Colors by
    class_name by default; if team_assignments is given, player detections
    are colored/labeled by team instead (goalkeeper, referee, and ball are
    unaffected -- team assignment only concerns players).

    Args:
        image: Image to draw on, modified in place.
        detections: All detections to render.
        scale: Resolution-relative scale factor.
        show_confidence: If True, append "{confidence:.2f}" to each label.
        team_assignments: Optional mapping of index into `detections` ->
            {"team_id": int, "confidence": float}, as produced by
            team_separation.assign_teams()["assignments"]. A player whose
            index has no entry here (e.g. dropped by that module's own
            confidence pre-filter) is labeled "player (no team)" rather than
            silently falling back to a plain "player" label, so it's visibly
            distinct from a player that team_separation actually assigned a
            team to. Pass None entirely to skip team-aware coloring and just
            color everything by class, as before.

    Returns:
        A tuple (counts, colors_by_label):
        - counts: number of detections per display label (class name, or
          "team 0" / "team 1" / "player (no team)" when team-colored).
        - colors_by_label: the color used for each display label, so callers
          can build a legend without re-deriving colors themselves.
    """
    height, width = image.shape[:2]
    box_thickness = max(1, round(2 * scale))
    font_scale = max(0.4, 0.5 * scale)
    text_thickness = max(1, round(1.5 * scale))
    margin = max(8, round(16 * scale))

    counts: Counter = Counter()
    colors_by_label: dict[str, tuple[int, int, int]] = {}

    for index, detection in enumerate(detections):
        if team_assignments is not None and detection.class_name == "player":
            entry = team_assignments.get(index)
            if entry is not None:
                display_label = f"team {entry['team_id']}"
                color = TEAM_COLORS.get(entry["team_id"], CLASS_COLORS["player"])
            else:
                display_label = "player (no team)"
                color = CLASS_COLORS["player"]
        else:
            display_label = detection.class_name
            color = CLASS_COLORS.get(detection.class_name, TEXT_COLOR)

        counts[display_label] += 1
        colors_by_label[display_label] = color

        x1, y1, x2, y2 = _bbox_corners(detection)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

        label = display_label
        if show_confidence:
            label = f"{label} {detection.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        box_width = text_width + 8
        box_height = text_height + baseline + 8

        # Prefer placing the label above the box; drop below if that would
        # clip off the top edge of the frame.
        label_x = min(max(x1, margin), width - box_width - margin)
        label_y = y1 - box_height
        if label_y < margin:
            label_y = y2

        _draw_label(
            image,
            label,
            (label_x, label_y),
            font_scale,
            text_thickness,
            text_color=(0, 0, 0),
            bg_color=color,
            alpha=DETECTION_LABEL_ALPHA,
            padding=4,
        )

    return counts, colors_by_label


def _draw_low_confidence_banner(image: np.ndarray, separation_score: float, scale: float) -> int:
    """Draw a full-width warning banner across the top of the image.

    Used when a supplied team_result flags low_confidence_frame -- the team
    colors below the banner should not be trusted. Solid (not
    semi-transparent) and full-width so it's impossible to miss, unlike the
    other overlays here.

    Args:
        image: Image to draw on, modified in place.
        separation_score: The frame's separation score, shown in the banner.
        scale: Resolution-relative scale factor.

    Returns:
        The banner height in pixels, so the legend and summary counts can be
        shifted below it and never overlap it.
    """
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, 0.85 * scale)
    thickness = max(2, round(2.5 * scale))
    padding = max(8, round(12 * scale))

    text = (
        f"LOW CONFIDENCE TEAM SPLIT (separation score {separation_score:.2f}) "
        "-- team colors below are not reliable"
    )
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    banner_height = text_height + baseline + 2 * padding

    cv2.rectangle(image, (0, 0), (width, banner_height), WARNING_BANNER_BG, -1)
    text_x = max(padding, (width - text_width) // 2)
    text_y = padding + text_height
    cv2.putText(
        image, text, (text_x, text_y), font, font_scale, WARNING_BANNER_TEXT, thickness, cv2.LINE_AA
    )
    return banner_height


def _build_foot_points_image(image_path: str, detections: list[Detection]) -> np.ndarray:
    """Build the foot-points annotated image without saving it.

    Factored out of debug_foot_points so debug_combined can compose it with
    the other two views without writing an intermediate file to disk.
    """
    image = _load_image(image_path)
    players = filter_players(detections)
    scale = _scale_factor(image)
    margin = max(8, round(16 * scale))

    marker_size = max(6, round(10 * scale))
    marker_thickness = max(1, round(2 * scale))

    for player in players:
        x, y = player.foot_point
        point = (int(round(x)), int(round(y)))
        cv2.drawMarker(
            image,
            point,
            FOOT_POINT_COLOR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=marker_size,
            thickness=marker_thickness,
            line_type=cv2.LINE_AA,
        )

    font_scale = max(0.4, 0.5 * scale)
    thickness = max(1, round(1.5 * scale))
    _draw_label(
        image, f"foot points: {len(players)}", (margin, margin), font_scale, thickness
    )
    return image


def _build_raw_detections_image(image_path: str, detections: list[Detection]) -> np.ndarray:
    """Build the raw-detections annotated image without saving it.

    Factored out of debug_raw_detections so debug_combined can compose it
    with the other two views without writing an intermediate file to disk.
    """
    image = _load_image(image_path)
    scale = _scale_factor(image)
    margin = max(8, round(16 * scale))

    counts, colors_by_label = _render_detections(image, detections, scale, show_confidence=False)

    entries = [(label, colors_by_label[label]) for label in sorted(counts)]
    _draw_legend(image, entries, scale, margin)
    _draw_summary_counts(image, counts, scale, margin)
    return image


def _build_annotated_frame_image(
    image_path: str, detections: list[Detection], team_result: dict | None = None
) -> np.ndarray:
    """Build the final annotated image without saving it.

    Factored out of debug_annotated_frame so debug_combined can compose it
    with the other two views without writing an intermediate file to disk.

    Args:
        image_path: Path to the original broadcast frame.
        detections: Full detection list, as returned by infer().
        team_result: Optional dict as returned by
            team_separation.assign_teams(). If given, players are colored by
            team instead of by class, and a warning banner is drawn if
            team_result["low_confidence_frame"] is True.
    """
    image = _load_image(image_path)
    scale = _scale_factor(image)
    margin = max(8, round(16 * scale))

    team_assignments = team_result["assignments"] if team_result is not None else None
    counts, colors_by_label = _render_detections(
        image, detections, scale, show_confidence=True, team_assignments=team_assignments
    )

    top_offset = 0
    if team_result is not None and team_result["low_confidence_frame"]:
        top_offset = _draw_low_confidence_banner(image, team_result["separation_score"], scale)

    entries = [(label, colors_by_label[label]) for label in sorted(counts)]
    _draw_legend(image, entries, scale, margin, top_offset=top_offset)
    _draw_summary_counts(image, counts, scale, margin, top_offset=top_offset)
    return image


def debug_foot_points(image_path: str, detections: list[Detection]) -> None:
    """Render only player foot points, to verify foot-point extraction.

    Deliberately minimal: no gridlines, no pitch overlay, no homography.
    This checks one thing only -- that each player's foot_point lands where
    their feet actually are in the raw image.

    Args:
        image_path: Path to the original broadcast frame.
        detections: Full detection list, as returned by infer(). Filtered
            internally to players via filter_players().

    Side effects:
        Writes an annotated JPEG to ./debug_output/foot_points_<timestamp>.jpg
        and prints the saved path.
    """
    image = _build_foot_points_image(image_path, detections)

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, f"foot_points_{_timestamp()}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved foot-point debug image to {output_path}")


def debug_raw_detections(image_path: str, detections: list[Detection]) -> None:
    """Render every raw detection (all four classes) with a legend and counts.

    This is the "did the model see everything it should have" check: every
    box the model produced, colored by class, with nothing filtered out.

    Args:
        image_path: Path to the original broadcast frame.
        detections: Full detection list, as returned by infer().

    Side effects:
        Writes an annotated JPEG to
        ./debug_output/raw_detections_<timestamp>.jpg and prints the saved
        path.
    """
    image = _build_raw_detections_image(image_path, detections)

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, f"raw_detections_{_timestamp()}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved raw detections debug image to {output_path}")


def debug_annotated_frame(
    image_path: str, detections: list[Detection], team_result: dict | None = None
) -> None:
    """Render the pixel-space "final" annotated frame: bbox + class/team + confidence.

    Colored by class_name (player/goalkeeper/referee/ball) by default. If
    team_result is given (the dict returned by team_separation.assign_teams()),
    players are colored by team_id instead -- goalkeeper, referee, and ball
    keep their normal class colors regardless, since team assignment only
    concerns players. A player team_separation excluded via its own
    confidence pre-filter is labeled "player (no team)" rather than silently
    reverting to a plain "player" label, so it's visibly distinct.

    If team_result["low_confidence_frame"] is True, a full-width warning
    banner is drawn across the top of the image and the legend/summary are
    shifted below it -- the team split for this frame should not be trusted,
    and that needs to be obvious at a glance, not just in a console log.

    Args:
        image_path: Path to the original broadcast frame.
        detections: Full detection list, as returned by infer().
        team_result: Optional dict as returned by
            team_separation.assign_teams(). Omit to color by class only, as
            before.

    Side effects:
        Writes an annotated JPEG to
        ./debug_output/annotated_final_<timestamp>.jpg and prints the saved
        path.
    """
    image = _build_annotated_frame_image(image_path, detections, team_result)

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, f"annotated_final_{_timestamp()}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved annotated final debug image to {output_path}")


def _make_section_header(width: int, text: str, scale: float) -> np.ndarray:
    """Build a solid black bar of the given width with centered white text.

    Used by debug_combined to separate the three stacked views so it's
    unambiguous which panel is which when scrolling through the combined
    image.

    Args:
        width: Bar width in pixels, matching the panels it separates.
        text: Text to center in the bar.
        scale: Resolution-relative scale factor.

    Returns:
        A BGR image of shape (height, width, 3).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, 0.7 * scale)
    thickness = max(1, round(2 * scale))
    padding = max(6, round(10 * scale))

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height = text_height + baseline + 2 * padding

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    text_x = max(padding, (width - text_width) // 2)
    text_y = padding + text_height
    cv2.putText(bar, text, (text_x, text_y), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    return bar


def debug_combined(
    image_path: str, detections: list[Detection], team_result: dict | None = None
) -> None:
    """Render all three debug views for one frame, stacked into a single image.

    Produces the same visuals as debug_foot_points, debug_raw_detections, and
    debug_annotated_frame, but grouped into one file instead of three -- useful
    when reviewing many frames, so each frame's diagnostics live in a single
    image instead of being scattered across a flat output folder.

    Args:
        image_path: Path to the original broadcast frame.
        detections: Full detection list, as returned by infer().
        team_result: Optional dict as returned by
            team_separation.assign_teams(), forwarded to the annotated-final
            panel. See debug_annotated_frame for details.

    Side effects:
        Writes a single annotated JPEG to
        ./debug_output/combined_<frame_name>_<timestamp>.jpg and prints the
        saved path.
    """
    foot_points_image = _build_foot_points_image(image_path, detections)
    raw_detections_image = _build_raw_detections_image(image_path, detections)
    annotated_image = _build_annotated_frame_image(image_path, detections, team_result)

    scale = _scale_factor(foot_points_image)
    width = foot_points_image.shape[1]

    annotated_title = (
        "ANNOTATED FINAL (team-colored)"
        if team_result is not None
        else "ANNOTATED FINAL (class-colored, not team-colored)"
    )
    panels = [
        ("FOOT POINTS", foot_points_image),
        ("RAW DETECTIONS", raw_detections_image),
        (annotated_title, annotated_image),
    ]

    rows = []
    for title, panel in panels:
        rows.append(_make_section_header(width, title, scale))
        rows.append(panel)
    combined = cv2.vconcat(rows)

    output_dir = _ensure_output_dir()
    frame_name = os.path.splitext(os.path.basename(image_path))[0]
    frame_name = "_".join(frame_name.split())
    output_path = os.path.join(output_dir, f"combined_{frame_name}_{_timestamp()}.jpg")
    cv2.imwrite(output_path, combined)
    print(f"Saved combined debug image to {output_path}")


if __name__ == "__main__":
    # Lazy import: keeps debug_tools.py usable without team_separation.py
    # installed/present for callers who only need class-based debugging.
    # team_separation.py (v1) is archived -- see archive/team_id_v1/ and
    # archive/README.md; production uses team_id_v2 instead.
    _archive_dir = os.path.join(_ROOT, "archive", "team_id_v1")
    if _archive_dir not in sys.path:
        sys.path.insert(0, _archive_dir)
    from team_separation import assign_teams

    # Replace this placeholder with a real broadcast frame path, or pass one
    # as a command-line argument: `python debug_tools.py path/to/frame.jpg`
    DEFAULT_IMAGE_PATH = "sample_frame.jpg"

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    all_detections = infer(image_path)

    debug_foot_points(image_path, all_detections)
    debug_raw_detections(image_path, all_detections)

    try:
        team_result = assign_teams(image_path, all_detections)
    except ValueError as exc:
        print(f"Team assignment skipped (falling back to class coloring): {exc}")
        team_result = None

    debug_annotated_frame(image_path, all_detections, team_result)
    debug_combined(image_path, all_detections, team_result)
