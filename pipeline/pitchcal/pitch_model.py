"""Geometric model of a FIFA-regulation football pitch.

Coordinate convention used throughout the package ("pitch coordinates"):

* Origin at one corner of the pitch, x along the long side (touchlines),
  y along the short side (goal lines).  Units are metres.
* x in [0, LENGTH] (default 105), y in [0, WIDTH] (default 68).
* The *canonical orientation* produced by the calibrator places the
  broadcast camera on the negative-y side, i.e. the touchline nearest the
  camera is y = 0 and the far touchline is y = WIDTH.  Which physical end
  of the stadium is x = 0 is not knowable from a single frame's geometry
  (the marking layout is 180-degree symmetric); canonicalisation makes the
  output *consistent* rather than absolutely oriented.

The pitch marking layout is exactly symmetric under the dihedral group
{identity, mirror-x, mirror-y, rotate-180}.  This module exposes those
symmetry transforms so the solver can reason about them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Standard FIFA dimensions (metres).
LENGTH: float = 105.0
WIDTH: float = 68.0
PENALTY_AREA_DEPTH: float = 16.5
PENALTY_AREA_HALF_WIDTH: float = 20.16  # 7.32/2 + 16.5
GOAL_AREA_DEPTH: float = 5.5
GOAL_AREA_HALF_WIDTH: float = 9.16  # 7.32/2 + 5.5
CIRCLE_RADIUS: float = 9.15
PENALTY_SPOT_DIST: float = 11.0
CORNER_ARC_RADIUS: float = 1.0

FamilyName = Literal["longitudinal", "transverse"]


@dataclass(frozen=True)
class TemplateLine:
    """One marked straight line of the pitch template.

    A template line is the *infinite* line at a fixed coordinate, plus the
    list of painted segments actually present on the pitch (e.g. the
    y = 13.84 line only exists inside the two penalty areas).
    """

    name: str
    family: FamilyName  # "longitudinal" = constant y, "transverse" = constant x
    position: float  # the constant coordinate (y for longitudinal, x for transverse)
    segments: tuple[tuple[float, float], ...]  # painted intervals along the free axis

    def endpoints(self, pitch: "PitchModel") -> tuple[np.ndarray, np.ndarray]:
        """Two well-separated points on the infinite line, in pitch coords."""
        lo, hi = (0.0, pitch.length) if self.family == "longitudinal" else (0.0, pitch.width)
        if self.family == "longitudinal":
            return np.array([lo, self.position]), np.array([hi, self.position])
        return np.array([self.position, lo]), np.array([self.position, hi])

    def sample(self, step: float = 0.5) -> np.ndarray:
        """Sample points along the painted segments only, shape (N, 2)."""
        pts = []
        for a, b in self.segments:
            ts = np.arange(a, b + 1e-9, step)
            if self.family == "longitudinal":
                pts.append(np.stack([ts, np.full_like(ts, self.position)], axis=1))
            else:
                pts.append(np.stack([np.full_like(ts, self.position), ts], axis=1))
        return np.concatenate(pts, axis=0)


@dataclass(frozen=True)
class TemplateArc:
    """A circular arc of the pitch template (full circles use span 2*pi)."""

    name: str
    center: tuple[float, float]
    radius: float
    theta0: float  # start angle, radians, CCW from +x
    theta1: float  # end angle (theta1 > theta0)

    def sample(self, step: float = 0.5) -> np.ndarray:
        n = max(8, int(np.ceil(self.radius * (self.theta1 - self.theta0) / step)))
        th = np.linspace(self.theta0, self.theta1, n)
        cx, cy = self.center
        return np.stack([cx + self.radius * np.cos(th), cy + self.radius * np.sin(th)], axis=1)


@dataclass
class PitchModel:
    """Full template: lines, circles/arcs, and the symmetry transforms."""

    length: float = LENGTH
    width: float = WIDTH
    lines: list[TemplateLine] = field(default_factory=list)
    arcs: list[TemplateArc] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lines:
            return
        L, W = self.length, self.width
        pw, gw = PENALTY_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH
        pd, gd = PENALTY_AREA_DEPTH, GOAL_AREA_DEPTH
        cy = W / 2.0
        self.lines = [
            TemplateLine("touch_near", "longitudinal", 0.0, ((0.0, L),)),
            TemplateLine("touch_far", "longitudinal", W, ((0.0, L),)),
            TemplateLine("pen_side_near", "longitudinal", cy - pw, ((0.0, pd), (L - pd, L))),
            TemplateLine("pen_side_far", "longitudinal", cy + pw, ((0.0, pd), (L - pd, L))),
            TemplateLine("goal_side_near", "longitudinal", cy - gw, ((0.0, gd), (L - gd, L))),
            TemplateLine("goal_side_far", "longitudinal", cy + gw, ((0.0, gd), (L - gd, L))),
            TemplateLine("goal_line_left", "transverse", 0.0, ((0.0, W),)),
            TemplateLine("goal_line_right", "transverse", L, ((0.0, W),)),
            TemplateLine("halfway", "transverse", L / 2.0, ((0.0, W),)),
            TemplateLine("pen_front_left", "transverse", pd, ((cy - pw, cy + pw),)),
            TemplateLine("pen_front_right", "transverse", L - pd, ((cy - pw, cy + pw),)),
            TemplateLine("goal_front_left", "transverse", gd, ((cy - gw, cy + gw),)),
            TemplateLine("goal_front_right", "transverse", L - gd, ((cy - gw, cy + gw),)),
        ]
        # Penalty arc: the part of the 9.15 m circle around the penalty spot
        # that lies outside the penalty area.
        half = float(np.arccos((pd - PENALTY_SPOT_DIST) / CIRCLE_RADIUS))
        self.arcs = [
            TemplateArc("center_circle", (L / 2.0, cy), CIRCLE_RADIUS, 0.0, 2.0 * np.pi),
            TemplateArc("pen_arc_left", (PENALTY_SPOT_DIST, cy), CIRCLE_RADIUS, -half, half),
            TemplateArc(
                "pen_arc_right",
                (L - PENALTY_SPOT_DIST, cy),
                CIRCLE_RADIUS,
                np.pi - half,
                np.pi + half,
            ),
            TemplateArc("corner_bl", (0.0, 0.0), CORNER_ARC_RADIUS, 0.0, np.pi / 2),
            TemplateArc("corner_tl", (0.0, W), CORNER_ARC_RADIUS, -np.pi / 2, 0.0),
            TemplateArc("corner_br", (L, 0.0), CORNER_ARC_RADIUS, np.pi / 2, np.pi),
            TemplateArc("corner_tr", (L, W), CORNER_ARC_RADIUS, np.pi, 3 * np.pi / 2),
        ]

    # ------------------------------------------------------------------ #
    # Family accessors used by the assignment search.
    # ------------------------------------------------------------------ #
    def family_lines(self, family: FamilyName) -> list[TemplateLine]:
        """Template lines of one family, sorted by their constant coordinate."""
        return sorted((l for l in self.lines if l.family == family), key=lambda l: l.position)

    def line_by_name(self, name: str) -> TemplateLine:
        for l in self.lines:
            if l.name == name:
                return l
        raise KeyError(name)

    # ------------------------------------------------------------------ #
    # Symmetries (3x3 homogeneous transforms acting on pitch coordinates).
    # ------------------------------------------------------------------ #
    def rot180(self) -> np.ndarray:
        """(x, y) -> (L - x, W - y): swaps ends AND camera side."""
        return np.array([[-1, 0, self.length], [0, -1, self.width], [0, 0, 1]], dtype=float)

    def mirror_x(self) -> np.ndarray:
        """(x, y) -> (L - x, y): swaps ends, keeps camera side.  Improper."""
        return np.array([[-1, 0, self.length], [0, 1, 0], [0, 0, 1]], dtype=float)

    def mirror_y(self) -> np.ndarray:
        """(x, y) -> (x, W - y): swaps camera side, keeps ends.  Improper."""
        return np.array([[1, 0, 0], [0, -1, self.width], [0, 0, 1]], dtype=float)

    def symmetries(self) -> list[np.ndarray]:
        return [np.eye(3), self.rot180(), self.mirror_x(), self.mirror_y()]

    # ------------------------------------------------------------------ #
    # Dense sampling of every painted curve (for scoring / rendering).
    # ------------------------------------------------------------------ #
    def sample_all(self, step: float = 0.5) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        for l in self.lines:
            out.append((l.name, l.sample(step)))
        for a in self.arcs:
            out.append((a.name, a.sample(step)))
        return out

    def circle_conic(self, which: str = "center_circle") -> np.ndarray:
        """3x3 conic matrix of a template circle: x^T C x = 0."""
        arc = next(a for a in self.arcs if a.name == which)
        cx, cy = arc.center
        r = arc.radius
        return np.array(
            [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy - r * r]]
        )
