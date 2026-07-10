"""tactical_map.py -- the M1 pipeline's final combined visualization.

One image per analyzed frame: the original broadcast frame with detections
annotated by team (left), a top-down FIFA-proportioned tactical map of the
recovered metric player positions (right), and an always-visible status
strip stating exactly how much the pipeline trusts its own output
(calibration source/confidence, team-separation quality, whether the
orientation correction fired).  The pipeline's design principle is honesty
about failure modes, so a frame with no usable calibration renders an
explicit "no calibration" panel rather than an empty or plausible-looking
map.

Purely additive: consumes :class:`frame_analysis.FrameAnalysis` and the
cached detections; modifies nothing upstream.  Colors are
``debug_tools``'s existing conventions (converted BGR -> matplotlib RGB),
with ONE deliberate divergence: players with ``team_id=None`` are drawn in
neutral gray here, whereas debug_tools falls back to the team-0/player
color -- in the *final* view an unassigned player must not silently look
like a team member.

Running the demo (the brasilscotland2 frame exercises the PnLCalib
fallback, which needs torch):

    ai-workspace/PnLCalib/.venv/bin/python tactical_map.py
"""

from __future__ import annotations

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")  # headless rendering; never require a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from debug_tools import CLASS_COLORS, TEAM_COLORS, _bbox_corners, _scale_factor  # noqa: E402
from detection import Detection  # noqa: E402
from frame_analysis import PITCH_LENGTH_M, PITCH_WIDTH_M, FrameAnalysis, PlayerRecord  # noqa: E402
from team_id_v2 import cached_infer  # noqa: E402

_DEFAULT_CACHE_DIR = os.path.join(_HERE, "detection_cache")


# ──────────────────────────────────────────────────────────────────────────
# Colors: debug_tools's BGR conventions, converted for matplotlib
# ──────────────────────────────────────────────────────────────────────────
def _bgr_to_mpl(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an OpenCV BGR uint8 color to a matplotlib RGB float triple.

    Args:
        bgr: (blue, green, red) in 0-255, as stored in debug_tools.

    Returns:
        (red, green, blue) in 0.0-1.0.
    """
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


#: Team colors, straight from debug_tools.TEAM_COLORS (team 0 azure blue,
#: team 1 magenta) so the final view matches every debug tool.
TEAM_COLORS_MPL: dict[int, tuple[float, float, float]] = {
    team: _bgr_to_mpl(bgr) for team, bgr in TEAM_COLORS.items()
}

#: Non-player classes keep their debug_tools class colors.
CLASS_COLORS_MPL: dict[str, tuple[float, float, float]] = {
    name: _bgr_to_mpl(bgr) for name, bgr in CLASS_COLORS.items()
}

#: Players with team_id=None (excluded crop / team pass failed) -- neutral
#: gray, deliberately NOT a team color (see module docstring).
NO_TEAM_COLOR: tuple[float, float, float] = (0.62, 0.62, 0.62)

#: Pitch rendering.
PITCH_GREEN = (0.16, 0.46, 0.22)
PITCH_LINE = (1.0, 1.0, 1.0)
PITCH_GREEN_DISABLED = (0.35, 0.37, 0.35)  # "no calibration" state

#: Status-strip background per calibration outcome (honest at a glance).
STRIP_BG: dict[str, tuple[float, float, float]] = {
    "pitchcal": (0.13, 0.35, 0.17),
    "pnlcalib_corrected": (0.13, 0.30, 0.38),
    "pnlcalib_rejected": (0.55, 0.12, 0.12),
    "failed": (0.55, 0.12, 0.12),
}


def _marker_color(player: PlayerRecord) -> tuple[float, float, float]:
    """Resolve a person's display color (shared by both panels).

    Args:
        player: one analyzed person record.

    Returns:
        Matplotlib RGB: team color for team-assigned players, neutral gray
        for unassigned players, class color for goalkeepers/referees.
    """
    if player.class_name == "player":
        if player.team_id is None:
            return NO_TEAM_COLOR
        return TEAM_COLORS_MPL.get(player.team_id, NO_TEAM_COLOR)
    return CLASS_COLORS_MPL.get(player.class_name, NO_TEAM_COLOR)


# ──────────────────────────────────────────────────────────────────────────
# Right panel: FIFA-proportioned top-down pitch
# ──────────────────────────────────────────────────────────────────────────
def _draw_pitch(
    ax: Axes,
    grass: tuple[float, float, float] = PITCH_GREEN,
    line_alpha: float = 1.0,
) -> None:
    """Draw a 105 x 68 m pitch in the corner-origin frame on ``ax``.

    Coordinates match PlayerRecord.pitch_position_m exactly: (0, 0) to
    (105, 68) metres, x along the length.  y=0 (the touchline nearest the
    camera in pitchcal's canonical frame) is drawn at the BOTTOM so the map
    reads like the broadcast view above it.

    Markings: touchlines/goal lines, halfway line, centre circle (9.15 m) +
    spot, penalty boxes (16.5 x 40.32 m), goal areas (5.5 x 18.32 m),
    penalty spots (11 m) + arcs (9.15 m, outside the box only), corner arcs
    (1 m).

    Args:
        ax: target axes; aspect is set to equal.
        grass: background color (dimmed for the no-calibration state).
        line_alpha: marking opacity (dimmed for the no-calibration state).
    """
    length, width = PITCH_LENGTH_M, PITCH_WIDTH_M
    lw = 1.4
    line = dict(color=PITCH_LINE, lw=lw, alpha=line_alpha, zorder=1)

    margin = 3.0
    ax.add_patch(
        Rectangle((-margin, -margin), length + 2 * margin, width + 2 * margin,
                  facecolor=grass, edgecolor="none", zorder=0)
    )
    ax.add_patch(Rectangle((0, 0), length, width, fill=False, **line))
    ax.plot([length / 2, length / 2], [0, width], **line)

    # Centre circle + spot.
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, **line))
    ax.add_patch(Circle((length / 2, width / 2), 0.30, color=PITCH_LINE, alpha=line_alpha, zorder=1))

    # Boxes, spots, penalty arcs -- both ends by symmetry.
    box_w, box_d = 40.32, 16.5  # penalty box: width (centred), depth
    goal_w, goal_d = 18.32, 5.5  # goal area
    spot_x, arc_r = 11.0, 9.15
    # Half-angle of the arc segment that lies OUTSIDE the penalty box:
    # cos(theta) = (box_d - spot_x) / arc_r.
    half_angle = float(np.degrees(np.arccos((box_d - spot_x) / arc_r)))
    for x_goal, direction in ((0.0, 1.0), (length, -1.0)):
        ax.add_patch(
            Rectangle((x_goal if direction > 0 else x_goal - box_d, (width - box_w) / 2),
                      box_d, box_w, fill=False, **line)
        )
        ax.add_patch(
            Rectangle((x_goal if direction > 0 else x_goal - goal_d, (width - goal_w) / 2),
                      goal_d, goal_w, fill=False, **line)
        )
        sx = x_goal + direction * spot_x
        ax.add_patch(Circle((sx, width / 2), 0.30, color=PITCH_LINE, alpha=line_alpha, zorder=1))
        centre_deg = 0.0 if direction > 0 else 180.0
        ax.add_patch(
            Arc((sx, width / 2), 2 * arc_r, 2 * arc_r,
                theta1=centre_deg - half_angle, theta2=centre_deg + half_angle,
                color=PITCH_LINE, lw=lw, alpha=line_alpha, zorder=1)
        )

    # Corner arcs (1 m quarter circles).
    for cx, cy, t1 in ((0, 0, 0), (length, 0, 90), (length, width, 180), (0, width, 270)):
        ax.add_patch(Arc((cx, cy), 2.0, 2.0, theta1=t1, theta2=t1 + 90,
                         color=PITCH_LINE, lw=lw, alpha=line_alpha, zorder=1))

    ax.set_xlim(-margin, length + margin)
    ax.set_ylim(-margin, width + margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_topdown_panel(ax: Axes, analysis: FrameAnalysis) -> None:
    """Render the tactical map (or the explicit no-calibration state).

    Args:
        ax: target axes.
        analysis: the frame analysis whose pitch_position_m values (already
            corner-origin metres) are plotted.
    """
    calibrated = analysis.calibration_source in ("pitchcal", "pnlcalib_corrected")
    if not calibrated:
        # Explicit failure state: dimmed pitch, loud message -- never an
        # empty plot that could pass for "no players detected".
        _draw_pitch(ax, grass=PITCH_GREEN_DISABLED, line_alpha=0.25)
        ax.text(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2 + 4,
                "NO CALIBRATION AVAILABLE\nFOR THIS FRAME",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white", zorder=3)
        reason = (
            f"calibration_source = {analysis.calibration_source}\n"
            + str(analysis.calibration_detail.get("pnlcalib_orientation_reason", "")).strip()
        ).strip()
        ax.text(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2 - 8, reason,
                ha="center", va="center", fontsize=8.5, color="white",
                alpha=0.9, wrap=True, zorder=3)
        return

    _draw_pitch(ax)
    for player in analysis.players:
        if player.pitch_position_m is None:
            continue
        x, y = player.pitch_position_m
        ax.add_patch(Circle((x, y), 1.1, facecolor=_marker_color(player),
                            edgecolor="white", lw=1.0, zorder=2))


# ──────────────────────────────────────────────────────────────────────────
# Left panel: annotated broadcast frame
# ──────────────────────────────────────────────────────────────────────────
def _draw_frame_panel(
    ax: Axes,
    frame_bgr: np.ndarray,
    analysis: FrameAnalysis,
    detections: list[Detection],
) -> None:
    """Render the broadcast frame with team-colored bounding boxes.

    Players are colored via :func:`_marker_color` (team colors / neutral
    gray), goalkeeper/referee/ball keep their debug_tools class colors.
    Box geometry comes from the detection list (PlayerRecord carries only
    foot points), joined on ``detection_index``.

    Args:
        ax: target axes.
        frame_bgr: original frame as loaded by cv2.
        analysis: per-person team assignments and indices.
        detections: the SAME detection list analyze_frame consumed (same
            cache), so indices line up with PlayerRecord.detection_index.
    """
    ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    by_index = {p.detection_index: p for p in analysis.players}
    lw = max(1.0, 1.4 * _scale_factor(frame_bgr))  # debug_tools's 720p-relative scaling
    for index, det in enumerate(detections):
        record = by_index.get(index)
        if record is not None:
            color = _marker_color(record)
        elif det.class_name in CLASS_COLORS_MPL:  # e.g. the ball
            color = CLASS_COLORS_MPL[det.class_name]
        else:
            continue
        x1, y1, x2, y2 = _bbox_corners(det)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, lw=lw))


def _legend_handles(analysis: FrameAnalysis) -> list[Line2D]:
    """Build legend entries for exactly the categories present in the frame.

    Args:
        analysis: the analyzed frame.

    Returns:
        Line2D proxy handles labelled team 0 / team 1 / player (no team) /
        goalkeeper / referee, in that order, only for categories that occur.
    """
    present: dict[str, tuple[float, float, float]] = {}
    for p in analysis.players:
        if p.class_name == "player":
            label = "player (no team)" if p.team_id is None else f"team {p.team_id}"
        else:
            label = p.class_name
        present.setdefault(label, _marker_color(p))
    order = ["team 0", "team 1", "player (no team)", "goalkeeper", "referee"]
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=present[label], markeredgecolor="white", label=label)
        for label in order
        if label in present
    ]


# ──────────────────────────────────────────────────────────────────────────
# Status strip
# ──────────────────────────────────────────────────────────────────────────
def _status_text(analysis: FrameAnalysis) -> str:
    """Compose the one-line honesty report shown above both panels.

    Args:
        analysis: the analyzed frame.

    Returns:
        A single readable line: calibration source (+ confidence when
        pitchcal produced one), team separation score + low-confidence
        flag, and pnlcalib_was_flipped whenever the fallback path ran.
    """
    detail = analysis.calibration_detail
    parts = [f"calibration: {analysis.calibration_source.upper()}"]
    conf = detail.get("pitchcal_confidence")
    if analysis.calibration_source == "pitchcal" and conf is not None:
        parts[-1] += f" (confidence {conf:.2f})"
    if "pnlcalib_was_flipped" in detail:
        parts.append(f"orientation flip applied: {detail['pnlcalib_was_flipped']}")
    if analysis.team_separation is not None:
        parts.append(f"team separation: {analysis.team_separation['separation_score']:.2f}")
        if analysis.team_separation["low_confidence_frame"]:
            parts.append("LOW-CONFIDENCE TEAM SPLIT")
    else:
        parts.append("team separation: FAILED")
    return "   |   ".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def render_tactical_map(
    analysis: FrameAnalysis,
    original_image_path: str,
    output_path: str,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> None:
    """Render the final combined view for one analyzed frame and save it.

    Left: the broadcast frame with detections boxed in team colors (neutral
    gray for team-less players; goalkeeper/referee/ball in their
    debug_tools class colors).  Right: a corner-origin 105 x 68 m tactical
    map of pitch_position_m, or an explicit "no calibration" state when
    calibration_source is "pnlcalib_rejected"/"failed".  Top: a status
    strip (color-coded by outcome) with calibration source/confidence,
    team-separation quality, and whether the orientation correction fired.

    Args:
        analysis: result of ``frame_analysis.analyze_frame`` for this frame.
        original_image_path: path to the SAME frame that was analyzed
            (bounding boxes are re-read from the detection cache, so the
            path must match the one given to analyze_frame).
        output_path: where to write the combined PNG/JPEG.
        cache_dir: detection cache directory; must be the one analyze_frame
            used so detection indices line up.

    Raises:
        FileNotFoundError: if the image is missing or undecodable.
    """
    frame_bgr = cv2.imread(original_image_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Image missing or undecodable: {original_image_path}")
    detections = cached_infer(original_image_path, cache_dir)

    fig = plt.figure(figsize=(19, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1, 16], width_ratios=[1.18, 1.0],
        left=0.015, right=0.985, top=0.985, bottom=0.02, wspace=0.03, hspace=0.06,
    )

    # Status strip spanning both panels.
    strip = fig.add_subplot(grid[0, :])
    strip.set_facecolor(STRIP_BG.get(analysis.calibration_source, STRIP_BG["failed"]))
    strip.set_xticks([])
    strip.set_yticks([])
    strip.text(0.5, 0.5, _status_text(analysis), transform=strip.transAxes,
               ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    ax_frame = fig.add_subplot(grid[1, 0])
    _draw_frame_panel(ax_frame, frame_bgr, analysis, detections)
    ax_frame.set_title(os.path.basename(original_image_path), fontsize=10, loc="left")

    ax_map = fig.add_subplot(grid[1, 1])
    _draw_topdown_panel(ax_map, analysis)
    ax_map.set_title("top-down tactical map (corner-origin metres, near touchline at bottom)",
                     fontsize=10, loc="left")

    handles = _legend_handles(analysis)
    if handles:
        ax_map.legend(handles=handles, loc="upper right", fontsize=9,
                      framealpha=0.9, borderpad=0.6)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Demo: one frame per calibration outcome (pitchcal + gated fallback)
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from frame_analysis import analyze_frame

    out_dir = os.path.join(_HERE, "debug_output")
    os.makedirs(out_dir, exist_ok=True)
    DEMO_FRAMES = [
        # pitchcal calibrates this one directly:
        os.path.join(_HERE, "Samples", "Screenshot From 2026-06-26 13-45-51.png"),
        # pitchcal refuses; PnLCalib fallback passes the orientation gate:
        os.path.join(_HERE, "Samples", "brasilscotland2.png"),
    ]
    for path in DEMO_FRAMES:
        fa = analyze_frame(path)
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        out = os.path.join(out_dir, f"tactical_{stem}.png")
        render_tactical_map(fa, path, out)
        print(f"{fa.calibration_source:20} -> {out}")
