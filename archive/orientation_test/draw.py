import cv2
import numpy as np

PALETTE: list[tuple[int, int, int]] = [
    (220,  80,  80),
    ( 80, 210,  80),
    ( 80,  80, 220),
    (220, 200,  50),
    (200,  80, 220),
    ( 50, 200, 220),
    (200, 130,  50),
    (130, 210,  50),
    ( 50, 130, 210),
    (210,  50, 130),
    ( 80, 170, 170),
    (170,  80, 170),
]


def _col(label: int) -> tuple[int, int, int]:
    return PALETTE[label % len(PALETTE)]


def _extend_line(
    centroid: np.ndarray, direction: np.ndarray, w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    cx, cy = float(centroid[0]), float(centroid[1])
    dx, dy = float(direction[0]), float(direction[1])
    ts: list[float] = []
    if abs(dx) > 1e-9:
        ts += [-cx / dx, (w - cx) / dx]
    if abs(dy) > 1e-9:
        ts += [-cy / dy, (h - cy) / dy]
    t_lo, t_hi = min(ts), max(ts)

    def clip(v: float, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(round(v))))

    x1 = clip(cx + t_lo * dx, 0, w - 1)
    y1 = clip(cy + t_lo * dy, 0, h - 1)
    x2 = clip(cx + t_hi * dx, 0, w - 1)
    y2 = clip(cy + t_hi * dy, 0, h - 1)
    return (x1, y1), (x2, y2)


def draw_output(
    img: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    clusters: list[dict],
    avg_angle: float,
    formation: str,
    debug: bool = False,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # PCA lines for passing clusters
    for c in clusters:
        if not c["passes"]:
            continue
        col = _col(c["label"])
        pt1, pt2 = _extend_line(c["centroid"], c["direction"], w, h)
        cv2.line(out, pt1, pt2, col, 1, cv2.LINE_AA)

        # Cluster annotation
        tx = int(c["centroid"][0]) + 6
        ty = int(c["centroid"][1]) - 8
        text = f'A={c["angle"]:.1f}  AR={c["aspect_ratio"]:.2f}'
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, col, 1, cv2.LINE_AA)

    # Player nodes colored by cluster
    for (x, y), lbl in zip(coords, labels):
        col = _col(int(lbl)) if lbl != -1 else (110, 110, 110)
        pt = (int(x), int(y))
        cv2.circle(out, pt, 6, col, -1)
        cv2.circle(out, pt, 6, (255, 255, 255), 1)

    # Weighted-average orientation line through image centre (white)
    cx_img, cy_img = w // 2, h // 2
    avg_rad = np.radians(avg_angle)
    avg_dir = np.array([np.cos(avg_rad), np.sin(avg_rad)])
    p1, p2 = _extend_line(np.array([cx_img, cy_img], dtype=float), avg_dir, w, h)
    cv2.line(out, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
    label_txt = f"pitch-x  {avg_angle:.1f}°"
    cv2.putText(out, label_txt, (cx_img + 8, cy_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label_txt, (cx_img + 8, cy_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Formation legend bar at bottom
    bar = np.zeros((44, w, 3), dtype=np.uint8)
    cv2.putText(bar, f"Formation: {formation}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([out, bar])

    if debug:
        out = _draw_debug_axes(out, coords, avg_angle)

    return out


def _draw_debug_axes(
    img: np.ndarray, coords: np.ndarray, avg_angle: float
) -> np.ndarray:
    out = img.copy()
    centroid = coords.mean(axis=0)
    cx, cy = int(centroid[0]), int(centroid[1])

    # Image axes from top-left (grey)
    org = (40, 40)
    ax = 80
    cv2.arrowedLine(out, org, (org[0] + ax, org[1]),
                    (140, 140, 140), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(out, "img-x", (org[0] + ax + 4, org[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.arrowedLine(out, org, (org[0], org[1] + ax),
                    (140, 140, 140), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(out, "img-y", (org[0] - 6, org[1] + ax + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Pitch axes from player centroid (yellow)
    avg_rad = np.radians(avg_angle)
    perp_rad = avg_rad + np.pi / 2
    pax = 100

    px_end = (int(cx + pax * np.cos(avg_rad)), int(cy + pax * np.sin(avg_rad)))
    cv2.arrowedLine(out, (cx, cy), px_end, (0, 215, 215), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(out, "pitch-x", (px_end[0] + 4, px_end[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 215), 1, cv2.LINE_AA)

    py_end = (int(cx + pax * np.cos(perp_rad)), int(cy + pax * np.sin(perp_rad)))
    cv2.arrowedLine(out, (cx, cy), py_end, (0, 190, 190), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(out, "pitch-y", (py_end[0] + 4, py_end[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 190, 190), 1, cv2.LINE_AA)

    # Observer arrow: centroid → top of image (green)
    h = out.shape[0]
    obs_tip = (cx, max(10, cy - 90))
    cv2.arrowedLine(out, (cx, cy), obs_tip, (0, 215, 0), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(out, "you", (obs_tip[0] + 5, obs_tip[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 0), 1, cv2.LINE_AA)

    return out


def draw_debug_schematic(
    coords: np.ndarray,
    labels: np.ndarray,
    rotated_coords: np.ndarray,
    avg_angle: float,
    img_shape: tuple,
) -> np.ndarray:
    CW, CH = 1100, 640
    canvas = np.full((CH, CW, 3), 238, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Pitch Orientation Debug — Image Space vs Pitch Space",
        (CW // 2 - 290, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA,
    )

    PAD = 18
    panel_w = (CW - 3 * PAD) // 2
    panel_h = CH - 60 - PAD

    _schematic_panel(
        canvas,
        x0=PAD, y0=55, pw=panel_w, ph=panel_h,
        title="Image Space  (raw pixel coords)",
        pts=coords, lab=labels,
        avg_angle=avg_angle, is_image_space=True,
    )
    _schematic_panel(
        canvas,
        x0=2 * PAD + panel_w, y0=55, pw=panel_w, ph=panel_h,
        title="Pitch Space  (rotation-corrected, −avg_angle applied)",
        pts=rotated_coords, lab=labels,
        avg_angle=avg_angle, is_image_space=False,
    )

    return canvas


def _schematic_panel(
    canvas: np.ndarray,
    x0: int, y0: int, pw: int, ph: int,
    title: str,
    pts: np.ndarray,
    lab: np.ndarray,
    avg_angle: float,
    is_image_space: bool,
) -> None:
    margin = 42
    uw = pw - 2 * margin
    uh = ph - 2 * margin

    # Background and border
    cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (218, 218, 218), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), 2)
    cv2.putText(canvas, title, (x0 + 6, y0 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)

    # Normalise coords, preserving aspect ratio
    x_min = pts[:, 0].min()
    y_min = pts[:, 1].min()
    x_rng = max(pts[:, 0].max() - x_min, 1.0)
    y_rng = max(pts[:, 1].max() - y_min, 1.0)
    scale = min(uw / x_rng, uh / y_rng)
    x_off = x0 + margin + (uw - x_rng * scale) / 2.0
    y_off = y0 + margin + (uh - y_rng * scale) / 2.0

    def sc(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(x_off + (x - x_min) * scale)),
            int(round(y_off + (y - y_min) * scale)),
        )

    # Player dots
    for (px_, py_), lbl in zip(pts, lab):
        col = _col(int(lbl)) if lbl != -1 else (130, 130, 130)
        cpt = sc(px_, py_)
        cv2.circle(canvas, cpt, 5, col, -1)
        cv2.circle(canvas, cpt, 5, (50, 50, 50), 1)

    # Axes drawn from player centroid (in panel coords)
    cx_d, cy_d = pts[:, 0].mean(), pts[:, 1].mean()
    ocx, ocy = sc(cx_d, cy_d)
    ax = 58

    if is_image_space:
        # img-x → right (grey)
        cv2.arrowedLine(canvas, (ocx, ocy), (ocx + ax, ocy),
                        (110, 110, 110), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, "img-x", (ocx + ax + 3, ocy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1, cv2.LINE_AA)
        # img-y → down (grey)
        cv2.arrowedLine(canvas, (ocx, ocy), (ocx, ocy + ax),
                        (110, 110, 110), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, "img-y", (ocx + 3, ocy + ax + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1, cv2.LINE_AA)
        # pitch-x at avg_angle (cyan)
        r = np.radians(avg_angle)
        pex = int(ocx + ax * np.cos(r))
        pey = int(ocy + ax * np.sin(r))
        cv2.arrowedLine(canvas, (ocx, ocy), (pex, pey),
                        (0, 160, 160), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, f"pitch-x ({avg_angle:.1f}°)",
                    (pex + 3, pey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "grey=image axes  |  cyan=recovered pitch-x",
                    (x0 + 6, y0 + ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1, cv2.LINE_AA)
    else:
        # pitch-x → right (cyan, already aligned)
        cv2.arrowedLine(canvas, (ocx, ocy), (ocx + ax, ocy),
                        (0, 160, 160), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, "pitch-x", (ocx + ax + 3, ocy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 140), 1, cv2.LINE_AA)
        # pitch-y → down (cyan)
        cv2.arrowedLine(canvas, (ocx, ocy), (ocx, ocy + ax),
                        (0, 160, 160), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, "pitch-y", (ocx + 3, ocy + ax + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 140), 1, cv2.LINE_AA)
        # In pitch space, img-x (1,0) derotated by -avg_angle → (cos(avg_angle), -sin(avg_angle))
        iex = int(ocx + ax * np.cos(np.radians(avg_angle)))
        iey = int(ocy - ax * np.sin(np.radians(avg_angle)))
        cv2.arrowedLine(canvas, (ocx, ocy), (iex, iey),
                        (110, 110, 110), 1, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, "img-x", (iex + 3, iey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "cyan=pitch axes  |  grey=original img-x (derotated)",
                    (x0 + 6, y0 + ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1, cv2.LINE_AA)
