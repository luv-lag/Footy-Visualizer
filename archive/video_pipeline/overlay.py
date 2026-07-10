"""Drawing and video writing."""

import numpy as np
import cv2

TEAM_A_COLOR = (255, 80, 80)    # red-ish (BGR)
TEAM_B_COLOR = (80, 80, 255)    # blue-ish (BGR)
BALL_COLOR = (0, 255, 255)      # yellow
EDGE_COLOR_A = (200, 60, 60)
EDGE_COLOR_B = (60, 60, 200)
NODE_RADIUS = 8
LEGEND_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _team_color(team: str) -> tuple[int, int, int]:
    return TEAM_A_COLOR if team == "a" else TEAM_B_COLOR


def _edge_color(team: str) -> tuple[int, int, int]:
    return EDGE_COLOR_A if team == "a" else EDGE_COLOR_B


def _draw_nodes_and_edges(
    frame: np.ndarray,
    players: list[dict],
    color: tuple,
    edge_color: tuple,
) -> None:
    pts = [(int(p["x"]), int(p["y"])) for p in players]

    # Intra-line edges: connect players within ~150px Y of each other
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][1] - pts[j][1]) < 150:
                cv2.line(frame, pts[i], pts[j], edge_color, 1, cv2.LINE_AA)

    for pt in pts:
        cv2.circle(frame, pt, NODE_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, NODE_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, ball: dict | None) -> None:
    if ball is None:
        return
    cx, cy = int(ball["x"]), int(ball["y"])
    cv2.drawMarker(frame, (cx, cy), BALL_COLOR, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)


def _draw_legend(
    frame: np.ndarray,
    formation_a: str,
    formation_b: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    legend = np.zeros((LEGEND_HEIGHT, w, 3), dtype=np.uint8)

    label_a = f"Team A: {formation_a}" if formation_a else "Team A"
    label_b = f"Team B: {formation_b}" if formation_b else "Team B"

    cv2.putText(legend, label_a, (10, 28), FONT, 0.75, TEAM_A_COLOR, 2, cv2.LINE_AA)
    cv2.putText(legend, label_b, (w // 2 + 10, 28), FONT, 0.75, TEAM_B_COLOR, 2, cv2.LINE_AA)

    return np.vstack([frame, legend])


def annotate_frame(
    frame: np.ndarray,
    formation_result: dict,
    ball: dict | None = None,
) -> np.ndarray:
    """Draw formation overlay on a single frame. Returns annotated frame."""
    out = frame.copy()

    _draw_nodes_and_edges(
        out,
        formation_result.get("team_a_players", []),
        TEAM_A_COLOR,
        EDGE_COLOR_A,
    )
    _draw_nodes_and_edges(
        out,
        formation_result.get("team_b_players", []),
        TEAM_B_COLOR,
        EDGE_COLOR_B,
    )
    _draw_ball(out, ball)

    out = _draw_legend(
        out,
        formation_result.get("team_a_formation", ""),
        formation_result.get("team_b_formation", ""),
    )
    return out


def open_writer(
    output_path: str,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h + LEGEND_HEIGHT))


def write_frame(writer: cv2.VideoWriter, annotated: np.ndarray) -> None:
    writer.write(annotated)
