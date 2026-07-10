"""Synthetic broadcast-frame renderer with exact ground-truth homographies.

Renders plausible broadcast views of a FIFA pitch (grass with mowing
stripes, white markings, crowd/ad-board clutter outside the pitch, player
blobs occluding lines, optional lens distortion) from a physically
parameterised camera, and returns the exact pitch->image homography.

Used by the validation suite: the full pipeline runs on these images and
its output is compared against ground truth in metres — including whether
the 180-degree orientation ambiguity was resolved correctly, which is
checkable here because the true camera side is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import DivisionModel, apply_h_w, to_h
from .pitch_model import PitchModel


@dataclass
class SyntheticFrame:
    """A rendered frame plus its ground truth."""

    image: np.ndarray  # BGR uint8
    H_gt: np.ndarray  # pitch -> *undistorted* image homography
    lam_gt: float  # division-model distortion coefficient (0 = none)
    camera_pos: np.ndarray  # world position (x, y, z) in pitch coords, metres
    label: str

    def gt_image_to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        """Map (distorted) pixel coords to pitch coords via ground truth."""
        dm = DivisionModel(self.lam_gt, (self.image.shape[1], self.image.shape[0]))
        und = dm.undistort(np.asarray(pts_px, dtype=float))
        Hinv = np.linalg.inv(self.H_gt)
        p, _ = apply_h_w(Hinv, und)
        return p


def camera_homography(
    cam_pos: np.ndarray,
    look_at: np.ndarray,
    focal_px: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Exact pitch-plane -> image homography for a pinhole camera.

    World frame = pitch coords with z up.  The camera is rolled so the
    image u-axis is horizontal in the world (zero roll).

    Returns H with p_img ~ H @ (x, y, 1).
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    z_cam = look_at - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(z_cam, up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)  # points "down" in world when looking ahead
    R = np.stack([x_cam, y_cam, z_cam])  # world -> camera rotation
    t = -R @ cam_pos
    w, h = image_size
    K = np.array([[focal_px, 0, w / 2.0], [0, focal_px, h / 2.0], [0, 0, 1.0]])
    P = K @ np.column_stack([R, t])  # 3x4
    H = P[:, [0, 1, 3]]  # pitch plane z = 0
    return H / np.linalg.norm(H)


def _draw_curves(
    img: np.ndarray,
    H: np.ndarray,
    pitch: PitchModel,
    dm: DivisionModel | None,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h_img, w_img = img.shape[:2]
    for _, pts in pitch.sample_all(step=0.25):
        proj, w = apply_h_w(H, pts)
        ok = w > 1e-9  # in front of the principal plane
        if dm is not None:
            proj = dm.distort(proj)
        # Draw as small consecutive segments, skipping breaks.
        vis = (
            ok
            & (proj[:, 0] > -200)
            & (proj[:, 0] < w_img + 200)
            & (proj[:, 1] > -200)
            & (proj[:, 1] < h_img + 200)
        )
        p = np.round(proj).astype(int)
        for i in range(len(p) - 1):
            if vis[i] and vis[i + 1] and np.abs(p[i + 1] - p[i]).max() < 80:
                cv2.line(img, tuple(p[i]), tuple(p[i + 1]), color, thickness, cv2.LINE_AA)


def render_frame(
    cam_pos: np.ndarray,
    look_at: np.ndarray,
    focal_px: float,
    image_size: tuple[int, int] = (1280, 720),
    lam: float = 0.0,
    n_players: int = 16,
    clutter: bool = True,
    line_thickness: int = 3,
    rng: np.random.Generator | None = None,
    label: str = "",
    pitch: PitchModel | None = None,
) -> SyntheticFrame:
    """Render one synthetic broadcast frame with exact ground truth."""
    rng = rng or np.random.default_rng(0)
    pitch = pitch or PitchModel()
    w, h = image_size
    H = camera_homography(cam_pos, look_at, focal_px, image_size)
    dm = DivisionModel(lam, image_size) if abs(lam) > 1e-9 else None

    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Crowd / stands background: noisy grey-brown texture.
    noise = rng.integers(40, 140, size=(h // 4, w // 4, 3), dtype=np.uint8)
    img[:] = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)

    # Grass polygon: project the pitch boundary (+ small apron).
    apron = 6.0
    boundary = []
    for x in np.linspace(-apron, pitch.length + apron, 80):
        boundary.append([x, -apron])
    for y in np.linspace(-apron, pitch.width + apron, 60):
        boundary.append([pitch.length + apron, y])
    for x in np.linspace(pitch.length + apron, -apron, 80):
        boundary.append([x, pitch.width + apron])
    for y in np.linspace(pitch.width + apron, -apron, 60):
        boundary.append([-apron, y])
    bpts = np.array(boundary)
    proj, wz = apply_h_w(H, bpts)
    if dm is not None:
        proj = dm.distort(proj)
    valid = wz > 1e-9
    if valid.sum() > 3:
        poly = np.round(proj[valid]).astype(int)
        poly = np.clip(poly, [-4 * w, -4 * h], [5 * w, 5 * h])
        cv2.fillPoly(img, [poly], (40, 120, 55))

    # Mowing stripes: alternate slightly lighter green bands along x.
    stripe = img.copy()
    for i in range(0, int(pitch.length / 5.25)):
        if i % 2 == 0:
            continue
        band = np.array(
            [
                [i * 5.25, -apron],
                [(i + 1) * 5.25, -apron],
                [(i + 1) * 5.25, pitch.width + apron],
                [i * 5.25, pitch.width + apron],
            ]
        )
        proj, wz = apply_h_w(H, band)
        if dm is not None:
            proj = dm.distort(proj)
        if np.all(wz > 1e-9):
            cv2.fillPoly(stripe, [np.round(proj).astype(int)], (52, 138, 66))
    img = cv2.addWeighted(img, 0.35, stripe, 0.65, 0)

    # Ad boards: bright rectangles just beyond the far touchline.
    if clutter:
        for x0 in np.arange(-5, pitch.length + 5, 12.0):
            band = np.array(
                [
                    [x0, pitch.width + apron + 0.2],
                    [x0 + 11.0, pitch.width + apron + 0.2],
                    [x0 + 11.0, pitch.width + apron + 2.2],
                    [x0, pitch.width + apron + 2.2],
                ]
            )
            proj, wz = apply_h_w(H, band)
            if dm is not None:
                proj = dm.distort(proj)
            if np.all(wz > 1e-9):
                color = tuple(int(c) for c in rng.integers(60, 255, 3))
                cv2.fillPoly(img, [np.round(proj).astype(int)], color)

    # Pitch markings.
    _draw_curves(img, H, pitch, dm, (245, 245, 245), line_thickness)

    # Players: ellipses standing on the pitch, some deliberately on lines.
    anchors = [
        np.array([pitch.length / 2, pitch.width / 2]),
        np.array([pitch.length / 2, 5.0]),
        np.array([16.5, pitch.width / 2]),
        np.array([pitch.length - 16.5, pitch.width / 2]),
    ]
    for i in range(n_players):
        if i < 4:
            pos = anchors[i] + rng.normal(0, 1.5, 2)
        else:
            pos = rng.uniform([2, 2], [pitch.length - 2, pitch.width - 2])
        foot, wz = apply_h_w(H, pos[None, :])
        if wz[0] <= 1e-9:
            continue
        f = foot[0]
        if dm is not None:
            f = dm.distort(f[None, :])[0]
        if not (0 <= f[0] < w and 0 <= f[1] < h):
            continue
        # Apparent height ~ 1.8 m: use local scale of the homography.
        head, _ = apply_h_w(H, (pos + [0, 0.6])[None, :])
        if dm is not None:
            head = dm.distort(head)
        px_per_m = float(np.linalg.norm(head[0] - f)) / 0.6
        hgt = max(6, int(1.8 * px_per_m))
        wid = max(3, hgt // 3)
        color = (30, 30, 160) if i % 2 else (200, 200, 200)
        cv2.ellipse(
            img,
            (int(f[0]), int(f[1] - hgt / 2)),
            (wid // 2, hgt // 2),
            0,
            0,
            360,
            color,
            -1,
        )

    # Sensor noise + slight blur.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    n = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float64) + n, 0, 255).astype(np.uint8)

    return SyntheticFrame(
        image=img, H_gt=H, lam_gt=lam, camera_pos=np.asarray(cam_pos, dtype=float), label=label
    )


def broadcast_camera_suite(
    n: int = 60,
    image_size: tuple[int, int] = (1280, 720),
    seed: int = 7,
    with_distortion: bool = False,
) -> list[SyntheticFrame]:
    """A varied suite of broadcast-like views for validation.

    Mix of wide halfway-line shots, penalty-area shots, corner-ish shots
    and tight midfield shots, from both sides of the pitch (so orientation
    canonicalisation is genuinely exercised) at varying height and zoom.
    """
    rng = np.random.default_rng(seed)
    pitch = PitchModel()
    frames: list[SyntheticFrame] = []
    kinds = ["halfway", "left_box", "right_box", "midfield", "corner"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        side = 1 if rng.random() < 0.5 else -1  # -1: camera at y < 0; +1: y > width
        cam_y = -rng.uniform(18, 45) if side < 0 else pitch.width + rng.uniform(18, 45)
        cam_x = 52.5 + rng.normal(0, 8)
        cam_z = rng.uniform(14, 32)
        if kind == "halfway":
            look = np.array([52.5 + rng.normal(0, 5), rng.uniform(20, 48), 0.0])
            f = rng.uniform(1100, 1900)
        elif kind == "left_box":
            look = np.array([rng.uniform(8, 20), rng.uniform(24, 44), 0.0])
            f = rng.uniform(1400, 2600)
        elif kind == "right_box":
            look = np.array([rng.uniform(85, 97), rng.uniform(24, 44), 0.0])
            f = rng.uniform(1400, 2600)
        elif kind == "corner":
            cx = 0.0 if rng.random() < 0.5 else 105.0
            look = np.array([abs(cx - rng.uniform(4, 14)), rng.uniform(8, 60), 0.0])
            f = rng.uniform(1500, 2600)
        else:  # tight midfield: often under-constrained on purpose
            look = np.array([rng.uniform(30, 75), rng.uniform(20, 48), 0.0])
            f = rng.uniform(2400, 4200)
        lam = float(rng.uniform(0.05, 0.22)) if with_distortion else 0.0
        frames.append(
            render_frame(
                np.array([cam_x, cam_y, cam_z]),
                look,
                f,
                image_size=image_size,
                lam=lam,
                rng=rng,
                label=f"{kind}_{'south' if side < 0 else 'north'}_{i:03d}",
                pitch=pitch,
            )
        )
    return frames
