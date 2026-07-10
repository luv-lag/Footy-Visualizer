"""
visualization.py — NEW: top-down metric pitch plot for approach_4 pipeline.

Draws detected players at their PnLCalib world positions on a to-scale
105 × 68 m pitch template. This is net-new code — no equivalent existed.

Coordinate frame (PnLCalib / FIFA standard):
  Origin at pitch centre.  X: length ±52.5 m.  Y: width ±34 m.

Plot axes:
  Horizontal → X (length, −52.5 left to +52.5 right).
  Vertical   → Y (width,  −34   bottom to +34   top).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── pitch geometry constants (FIFA standard) ──────────────────────────────────
_PITCH_LEN  = 105.0
_PITCH_WID  =  68.0
_HL         = _PITCH_LEN / 2   # 52.5
_HW         = _PITCH_WID / 2   # 34.0

# penalty area: 16.5m from goal line, 40.32m wide (centred)
_PA_DEPTH   = 16.5
_PA_HALF_W  = 20.16

# goal area: 5.5m from goal line, 18.32m wide
_GA_DEPTH   =  5.5
_GA_HALF_W  =  9.16

# penalty spot: 11m from goal line
_PEN_DIST   = 11.0

# centre circle radius
_CC_RAD     =  9.15


# ── team colours ──────────────────────────────────────────────────────────────
_TEAM_COLORS = {0: "#e74c3c", 1: "#3498db", -1: "#95a5a6"}  # red, blue, grey
_OOB_COLOR   = "#f39c12"  # orange X for out-of-bounds


def _draw_pitch(ax: plt.Axes) -> None:
    """Draw a minimal but complete top-down pitch template on ax."""
    lw   = 1.5
    col  = "white"
    rect = plt.Rectangle((-_HL, -_HW), _PITCH_LEN, _PITCH_WID,
                          fill=False, edgecolor=col, linewidth=lw)
    ax.add_patch(rect)

    # halfway line
    ax.plot([0, 0], [-_HW, _HW], color=col, linewidth=lw)

    # centre circle
    cc = mpatches.Circle((0, 0), _CC_RAD, fill=False, edgecolor=col, linewidth=lw)
    ax.add_patch(cc)
    ax.plot(0, 0, "o", color=col, markersize=3)

    for sign in (-1, 1):
        gx = sign * _HL   # goal-line X

        # penalty area
        ax.add_patch(plt.Rectangle(
            (gx - sign * _PA_DEPTH, -_PA_HALF_W),
            sign * _PA_DEPTH, 2 * _PA_HALF_W,
            fill=False, edgecolor=col, linewidth=lw,
        ))

        # goal area
        ax.add_patch(plt.Rectangle(
            (gx - sign * _GA_DEPTH, -_GA_HALF_W),
            sign * _GA_DEPTH, 2 * _GA_HALF_W,
            fill=False, edgecolor=col, linewidth=lw,
        ))

        # penalty spot
        ax.plot(gx - sign * _PEN_DIST, 0, "o", color=col, markersize=3)

        # penalty arc (arc of circle centred on penalty spot, outside PA)
        spot_x = gx - sign * _PEN_DIST
        arc = mpatches.Arc(
            (spot_x, 0), 2 * _CC_RAD, 2 * _CC_RAD,
            angle=0,
            theta1=(-40 if sign == 1 else 140),
            theta2=(40  if sign == 1 else 220),
            color=col, linewidth=lw,
        )
        ax.add_patch(arc)


def plot_topdown(
    projected_players,          # list[ProjectedPlayer]
    output_path: str,
    title: str = "Player positions — top-down (PnLCalib world frame)",
) -> None:
    """Render a metric top-down pitch plot and save to output_path.

    Parameters
    ----------
    projected_players : list of ProjectedPlayer from projection.project_players().
    output_path       : destination PNG path (parent dirs created automatically).
    title             : figure title.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#2d7a2d")   # grass green background
    ax.set_facecolor("#2d7a2d")

    _draw_pitch(ax)

    n_valid = 0
    n_oob   = 0

    for p in projected_players:
        if p.world_xy is None:
            continue
        xw, yw = p.world_xy
        if p.oob:
            ax.plot(xw, yw, "x", color=_OOB_COLOR, markersize=10, markeredgewidth=2)
            n_oob += 1
        else:
            color = _TEAM_COLORS.get(p.team, _TEAM_COLORS[-1])
            ax.plot(xw, yw, "o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.8)
            n_valid += 1

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    teams_present = {p.team for p in projected_players if p.world_xy is not None and not p.oob}
    for t in sorted(teams_present):
        legend_handles.append(
            mpatches.Patch(facecolor=_TEAM_COLORS.get(t, _TEAM_COLORS[-1]),
                           edgecolor="white",
                           label=f"Team {t}  (n={sum(1 for p in projected_players if p.team==t and p.world_xy is not None and not p.oob)})")
        )
    if n_oob:
        legend_handles.append(
            mpatches.Patch(facecolor=_OOB_COLOR, label=f"Out-of-bounds (n={n_oob})")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=9, framealpha=0.75)

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-_HL - 3, _HL + 3)
    ax.set_ylim(-_HW - 3, _HW + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("X — length axis (m)", color="white")
    ax.set_ylabel("Y — width axis (m)",  color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    n_total = sum(1 for p in projected_players if p.world_xy is not None)
    ax.set_title(
        f"{title}\n{n_valid} in-bounds  |  {n_oob} out-of-bounds  |  "
        f"{len(projected_players) - n_total} projection failures",
        color="white", fontsize=11, pad=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] top-down plot saved → {output_path}")
