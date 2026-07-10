"""Projective-geometry utilities: homogeneous helpers, DLT, lens model,
handedness / orientation canonicalisation.

Conventions:

* ``H`` always denotes the pitch->image homography (pitch coords in metres
  to *undistorted* pixel coords).  The inverse maps image->pitch.
* Lines are homogeneous 3-vectors ``l`` with ``l . p = 0`` for homogeneous
  points ``p`` on the line.
* Under ``p' = H p``, lines map as ``l' = H^-T l``.

Handedness: for any physical camera above the pitch plane producing a
non-mirrored image (image v-axis pointing down), the pitch->image mapping
is orientation-reversing: ``sign(det J) = sign(det H) * sign(w)^3 < 0`` at
every pitch point, where ``w`` is the third homogeneous output coordinate.
(x-then-y on the pitch is counter-clockwise seen from above; a camera above
the plane preserves that, and the image's downward v-axis flips it.)  A
solution violating this contains an odd number of mirror symmetries and is
physically impossible, so the solver rejects it outright.  This is verified
against rendered ground truth in the test-suite.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------- #
# Homogeneous helpers
# ---------------------------------------------------------------------- #
def to_h(pts: np.ndarray) -> np.ndarray:
    """(N, 2) -> (N, 3) homogeneous."""
    pts = np.asarray(pts, dtype=float)
    return np.concatenate([pts, np.ones((len(pts), 1))], axis=1)


def from_h(pts_h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(N, 3) -> (N, 2); points with |w| < eps become +/-inf."""
    w = pts_h[:, 2:3].copy()
    w[np.abs(w) < eps] = np.sign(w[np.abs(w) < eps]) * eps + (w[np.abs(w) < eps] == 0) * eps
    return pts_h[:, :2] / w


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) inhomogeneous points."""
    return from_h(to_h(pts) @ H.T)


def apply_h_w(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`apply_h` but also returns the w coordinate per point."""
    ph = to_h(pts) @ H.T
    return from_h(ph), ph[:, 2]


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two inhomogeneous 2D points."""
    return np.cross(np.array([p[0], p[1], 1.0]), np.array([q[0], q[1], 1.0]))


def normalize_line(l: np.ndarray) -> np.ndarray:
    """Scale a line so (a, b) is unit length (point-line distance in pixels)."""
    n = float(np.hypot(l[0], l[1]))
    return l / n if n > 1e-12 else l


def intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Homogeneous intersection point of two lines."""
    return np.cross(l1, l2)


def point_line_distance(l: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perpendicular distances from (N, 2) points to a homogeneous line."""
    l = normalize_line(np.asarray(l, dtype=float))
    return np.abs(to_h(pts) @ l)


# ---------------------------------------------------------------------- #
# Hartley normalisation
# ---------------------------------------------------------------------- #
def normalizing_transform(pts: np.ndarray) -> np.ndarray:
    """Similarity T mapping points to zero mean / sqrt(2) RMS distance."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


# ---------------------------------------------------------------------- #
# DLT from mixed point / line / vanishing-point correspondences
# ---------------------------------------------------------------------- #
def dlt_homography(
    point_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    line_pairs: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] | None = None,
    vp_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray | None, float]:
    """Solve pitch->image H from mixed correspondences via normalised DLT.

    Args:
        point_pairs: list of (pitch_xy (2,), image_xy (2,)) correspondences.
        line_pairs: list of (image_line (3,), (pitch_pt_a, pitch_pt_b))
            where the two pitch points span the template line; each yields
            two rows ``l^T H P = 0``.
        vp_pairs: list of (pitch_direction (2,), image_point_h (3,)):
            a vanishing-point correspondence -- the pitch point at infinity
            ``(dx, dy, 0)`` must map to the given homogeneous image point.

    Returns:
        (H, residual) with residual the smallest singular value of the
        normalised design matrix (0 for an exact solution), or (None, inf)
        if under-constrained / degenerate.
    """
    point_pairs = point_pairs or []
    line_pairs = line_pairs or []
    vp_pairs = vp_pairs or []

    img_pts = [ip for _, ip in point_pairs]
    pitch_pts = [pp for pp, _ in point_pairs]
    for _, (pa, pb) in line_pairs:
        pitch_pts += [pa, pb]
    if len(img_pts) >= 2:
        T = normalizing_transform(np.array(img_pts))
    else:
        T = np.diag([1 / 500.0, 1 / 500.0, 1.0])
    S = normalizing_transform(np.array(pitch_pts)) if len(pitch_pts) >= 2 else np.eye(3)
    Tinv = np.linalg.inv(T)
    Tit = Tinv.T

    rows: list[np.ndarray] = []
    for P, p in point_pairs:
        Pn = S @ np.array([P[0], P[1], 1.0])
        pn = T @ np.array([p[0], p[1], 1.0])
        x, y, w = pn
        rows.append(np.concatenate([np.zeros(3), -w * Pn, y * Pn]))
        rows.append(np.concatenate([w * Pn, np.zeros(3), -x * Pn]))
    for l_img, (pa, pb) in line_pairs:
        ln = normalize_line(Tit @ np.asarray(l_img, dtype=float))
        for P in (pa, pb):
            Pn = S @ np.array([P[0], P[1], 1.0])
            rows.append(np.concatenate([ln[0] * Pn, ln[1] * Pn, ln[2] * Pn]))
    for d, vp in vp_pairs:
        Pn = S @ np.array([d[0], d[1], 0.0])  # direction: no translation applied... S scales
        pn = T @ np.asarray(vp, dtype=float)
        x, y, w = pn
        rows.append(np.concatenate([np.zeros(3), -w * Pn, y * Pn]))
        rows.append(np.concatenate([w * Pn, np.zeros(3), -x * Pn]))

    if len(rows) < 7:
        return None, float("inf")
    A = np.stack(rows)
    try:
        _, sv, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None, float("inf")
    # Degeneracy check: the nullspace must be exactly one-dimensional.
    # With 8 rows the 8th singular value indicates rank deficiency; with
    # more rows the second-smallest of the 9 does.
    gap = float(sv[7]) if len(sv) >= 8 else 0.0
    if len(sv) >= 9:
        gap = float(sv[-2])
    if gap < 1e-7 * float(sv[0]):
        return None, float("inf")
    Hn = Vt[-1].reshape(3, 3)
    resid = float(sv[-1]) if len(sv) >= 9 else 0.0
    H = Tinv @ Hn @ S
    if abs(np.linalg.det(H)) < 1e-12:
        return None, float("inf")
    return H / np.linalg.norm(H), resid


def dlt_homography_family(
    point_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    line_pairs: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] | None = None,
    vp_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    degenerate_ratio: float = 5e-3,
    n_samples: int = 48,
    score_fn=None,
) -> list[tuple[np.ndarray, float]]:
    """Like :func:`dlt_homography`, but under-determined-by-one systems
    return a *sampled one-parameter family* of solutions instead of
    nothing / an arbitrary nullspace vector.

    Several pitch configurations (e.g. three concurrent transverse lines +
    one touchline, or conic points on their own carrier line) constrain a
    homography up to an exact one-parameter subgroup.  Downstream global
    scoring against the observed marking pixels can select the member, so
    the family is worth surfacing.

    Args:
        score_fn: optional ``H -> float`` (lower is better, inf = invalid).
            The family angle parameterisation is extremely non-uniform, so
            coarse samples alone can miss the true member; when a score is
            available the best coarse sample is polished by golden-section
            search over the family angle.

    Returns a list of (H, resid): one entry for a well-posed system, a
    handful of family members for a rank-deficient one, empty if hopeless.
    """
    point_pairs = point_pairs or []
    line_pairs = line_pairs or []
    vp_pairs = vp_pairs or []
    # Reuse the single-solution path first (handles normalisation).
    img_pts = [ip for _, ip in point_pairs]
    pitch_pts = [pp for pp, _ in point_pairs]
    for _, (pa, pb) in line_pairs:
        pitch_pts += [pa, pb]
    if len(img_pts) >= 2:
        T = normalizing_transform(np.array(img_pts))
    else:
        T = np.diag([1 / 500.0, 1 / 500.0, 1.0])
    S = normalizing_transform(np.array(pitch_pts)) if len(pitch_pts) >= 2 else np.eye(3)
    Tinv = np.linalg.inv(T)
    Tit = Tinv.T

    rows: list[np.ndarray] = []
    for P, p in point_pairs:
        Pn = S @ np.array([P[0], P[1], 1.0])
        pn = T @ np.array([p[0], p[1], 1.0])
        x, y, w = pn
        rows.append(np.concatenate([np.zeros(3), -w * Pn, y * Pn]))
        rows.append(np.concatenate([w * Pn, np.zeros(3), -x * Pn]))
    for l_img, (pa, pb) in line_pairs:
        ln = normalize_line(Tit @ np.asarray(l_img, dtype=float))
        for P in (pa, pb):
            Pn = S @ np.array([P[0], P[1], 1.0])
            rows.append(np.concatenate([ln[0] * Pn, ln[1] * Pn, ln[2] * Pn]))
    for d, vp in vp_pairs:
        Pn = S @ np.array([d[0], d[1], 0.0])
        pn = T @ np.asarray(vp, dtype=float)
        x, y, w = pn
        rows.append(np.concatenate([np.zeros(3), -w * Pn, y * Pn]))
        rows.append(np.concatenate([w * Pn, np.zeros(3), -x * Pn]))
    if len(rows) < 7:
        return []
    A = np.stack(rows)
    try:
        _, sv, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return []
    n_sv = len(sv)
    gap = float(sv[7]) if n_sv >= 8 else 0.0
    if n_sv >= 9:
        gap = float(sv[-2])
    resid = float(sv[-1]) if n_sv >= 9 else 0.0

    def build(hvec: np.ndarray) -> np.ndarray | None:
        H = Tinv @ hvec.reshape(3, 3) @ S
        if abs(np.linalg.det(H)) < 1e-12:
            return None
        return H / np.linalg.norm(H)

    if gap >= degenerate_ratio * float(sv[0]):
        H = build(Vt[-1])
        return [(H, resid)] if H is not None else []
    # Two-dimensional nullspace: sample the projective pencil of solutions.
    v9, v8 = Vt[-1], Vt[-2]

    def member(th: float) -> np.ndarray | None:
        return build(np.cos(th) * v9 + np.sin(th) * v8)

    thetas = np.linspace(0.0, np.pi, n_samples, endpoint=False)
    if score_fn is None:
        out = [(member(th), max(resid, gap)) for th in thetas]
        return [(H, r) for H, r in out if H is not None]

    def sc(th: float) -> float:
        H = member(th)
        if H is None:
            return float("inf")
        s = score_fn(H)
        return s if np.isfinite(s) else float("inf")

    scores = np.array([sc(th) for th in thetas])
    if not np.isfinite(scores).any():
        return []
    order = np.argsort(scores)
    out = []
    step = np.pi / n_samples
    for i in order[:3]:  # polish the best few coarse samples
        if not np.isfinite(scores[i]):
            break
        a, b = thetas[i] - step, thetas[i] + step
        # Golden-section search on the family angle.
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = sc(c), sc(d)
        for _ in range(18):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = sc(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = sc(d)
        th_best = (a + b) / 2.0
        H = member(th_best)
        if H is not None and np.isfinite(sc(th_best)):
            out.append((H, max(resid, gap)))
    return out


# ---------------------------------------------------------------------- #
# Conics
# ---------------------------------------------------------------------- #
def ellipse_to_conic(cx: float, cy: float, a: float, b: float, angle_rad: float) -> np.ndarray:
    """3x3 conic matrix of an ellipse (semi-axes a, b; rotation angle)."""
    R = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
    )
    D = np.diag([1.0 / (a * a), 1.0 / (b * b)])
    M = R @ D @ R.T
    C = np.zeros((3, 3))
    C[:2, :2] = M
    c = np.array([cx, cy])
    C[:2, 2] = -M @ c
    C[2, :2] = -M @ c
    C[2, 2] = c @ M @ c - 1.0
    return C / np.linalg.norm(C)


def transform_conic(C_src: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Conic of the image of ``C_src`` under point map ``p' = H p``."""
    Hinv = np.linalg.inv(H)
    C = Hinv.T @ C_src @ Hinv
    return C / np.linalg.norm(C)


def line_conic_intersections(l: np.ndarray, C: np.ndarray) -> np.ndarray | None:
    """Real intersection points (2, 2) of a line and a conic, else None.

    Numerically conditioned by parameterising the line from the foot of
    the perpendicular dropped from the conic's centre, so parameter values
    stay at the conic's own scale even for lines given in ill-scaled
    homogeneous form (e.g. polars of near-infinite points).
    """
    l = normalize_line(np.asarray(l, dtype=float))
    a, b, c = l
    d = np.array([b, -a, 0.0])
    # Conic centre (finite for ellipses); fall back to axis intercept.
    try:
        ctr = np.linalg.solve(C[:2, :2], -C[:2, 2])
        if not np.isfinite(ctr).all():
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        ctr = np.array([-c * a, -c * b])
    # Foot of perpendicular from the centre onto the line.
    dist = a * ctr[0] + b * ctr[1] + c
    foot = ctr - dist * np.array([a, b])
    p0 = np.array([foot[0], foot[1], 1.0])
    A = d @ C @ d
    B = 2.0 * (p0 @ C @ d)
    Cc = p0 @ C @ p0
    disc = B * B - 4 * A * Cc
    if disc < 0 or abs(A) < 1e-18:
        return None
    ts = np.array([(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)])
    pts = np.stack([p0 + t * d for t in ts])
    return from_h(pts)


def tangent_lines_from_point(
    v: np.ndarray, C: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Tangent lines to conic C through homogeneous point v, with contacts.

    Uses pole-polar duality: the polar of v intersects C in the two contact
    points p, and the tangent at p is C p.  Returns ([line1, line2],
    contacts (2, 2)) with line_i tangent at contacts[i], or None when v is
    inside the conic.  The contact points matter: tangent lines alone leave
    a one-parameter elation family of homographies unresolved.
    """
    v = np.asarray(v, dtype=float)
    v = v / max(np.abs(v).max(), 1e-300)
    polar = C @ v  # polar line of v: contains both contact points
    pts = line_conic_intersections(polar, C)
    if pts is None:
        return None
    lines = [normalize_line(C @ np.array([p[0], p[1], 1.0])) for p in pts]
    return lines, pts


# ---------------------------------------------------------------------- #
# Handedness and canonical orientation
# ---------------------------------------------------------------------- #
def jacobian_det_sign(H: np.ndarray, pts_pitch: np.ndarray) -> np.ndarray:
    """Sign of det(Jacobian) of the pitch->image map at pitch points.

    det J = det(H) / w^3 with w the third homogeneous output coordinate.
    """
    w = (to_h(pts_pitch) @ H.T)[:, 2]
    return np.sign(np.linalg.det(H)) * np.sign(w) ** 3


def proper_view_violation(
    H: np.ndarray,
    length: float,
    width: float,
    image_size: tuple[float, float] | None = None,
) -> str | None:
    """Why H is NOT orientation-valid for a real camera, or None if it is.

    On tight shots part of the pitch legitimately lies behind the camera's
    principal plane, so the checks apply to the pitch points that project
    *into the frame*: (a) they must share one w sign (all in front), and
    (b) the mapping must be orientation-reversing there (non-mirrored view
    from above; see module docstring).  With ``image_size`` omitted a
    generous default frame is assumed.

    Returns ``None`` when the view is proper, else a human-readable
    description of the physical impossibility.
    """
    gx, gy = np.meshgrid(np.linspace(1, length - 1, 15), np.linspace(1, width - 1, 11))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    ph = to_h(pts) @ H.T
    w = ph[:, 2]
    safe = np.abs(w) > 1e-12
    if not safe.any():
        return "degenerate mapping: every probed pitch point maps to infinity"
    proj = ph[safe, :2] / w[safe, None]
    w = w[safe]
    if image_size is None:
        fw, fh = 4000.0, 2500.0
    else:
        fw, fh = float(image_size[0]), float(image_size[1])
    lo = np.array([-fw * 0.1, -fh * 0.1])
    hi = np.array([fw * 1.1, fh * 1.1])
    vis = (
        (proj[:, 0] > lo[0])
        & (proj[:, 0] < hi[0])
        & (proj[:, 1] > lo[1])
        & (proj[:, 1] < hi[1])
    )
    if vis.sum() < 4:
        return "fewer than 4 pitch points project into the frame"
    wv = w[vis]
    if np.any(wv > 0) and np.any(wv < 0):
        return (
            "visible pitch points lie on both sides of the camera's "
            "principal plane (impossible for a physical view)"
        )
    s = np.sign(wv[0])
    # det J = det(H) / w^3; require det J < 0 at visible points.
    if not bool(np.sign(np.linalg.det(H)) * s < 0):
        return (
            "mirror-image solution: the mapping is orientation-preserving, "
            "which no physical camera above the pitch can produce"
        )
    return None


def is_proper_view(
    H: np.ndarray,
    length: float,
    width: float,
    image_size: tuple[float, float] | None = None,
) -> bool:
    """True iff H is orientation-valid for a real camera above the pitch.

    Boolean wrapper around :func:`proper_view_violation` (one shared
    implementation; see there for the physics).
    """
    return proper_view_violation(H, length, width, image_size) is None


def near_side_is_low_y(H: np.ndarray, length: float, width: float) -> bool:
    """True iff the y=0 touchline is nearer the camera than y=width.

    Nearness is measured by local image scale |det J| = |det H| / |w|^3:
    the near side of the pitch projects with larger scale (smaller |w|).
    Uses several probe pairs across the pitch and majority-votes over the
    ones lying in front of the camera, so tight shots where part of the
    pitch is behind the principal plane still vote correctly.
    """
    votes = 0
    total = 0
    for fx in (0.2, 0.5, 0.8):
        p = np.array([[length * fx, width * 0.1], [length * fx, width * 0.9]])
        w = (to_h(p) @ H.T)[:, 2]
        if np.any(np.abs(w) < 1e-12) or w[0] * w[1] <= 0:
            continue  # one probe behind the camera: skip this pair
        total += 1
        votes += int(abs(w[0]) < abs(w[1]))
    if total == 0:  # degenerate; fall back to the centre pair regardless
        p = np.array([[length / 2, width * 0.1], [length / 2, width * 0.9]])
        w = (to_h(p) @ H.T)[:, 2]
        return bool(abs(w[0]) < abs(w[1]))
    return votes * 2 > total


def canonicalize(H: np.ndarray, length: float, width: float) -> tuple[np.ndarray, bool]:
    """Compose H with rot180 if needed so the camera sits on the y<0 side.

    Returns (H_canonical, was_flipped).  H must already be a proper view
    (mirror solutions must have been rejected earlier).
    """
    if near_side_is_low_y(H, length, width):
        return H, False
    R = np.array([[-1, 0, length], [0, -1, width], [0, 0, 1]], dtype=float)
    Hc = H @ R
    return Hc / np.linalg.norm(Hc), True


# ---------------------------------------------------------------------- #
# One-parameter division-model lens distortion
# ---------------------------------------------------------------------- #
class DivisionModel:
    """Fitzgibbon single-parameter division model.

    Undistortion: with n = (p - c) / f0 (f0 = image diagonal / 2),
    ``undistorted = c + f0 * n / (1 + lam * |n|^2)``.

    lam > 0 corrects barrel (fisheye) distortion.  lam = 0 is the identity.
    The centre defaults to the image centre but may be offset -- virtual
    pan-tilt-zoom crops of panoramic rigs have off-centre distortion.
    """

    def __init__(
        self,
        lam: float,
        image_size: tuple[int, int],
        center: tuple[float, float] | None = None,
    ) -> None:
        self.lam = float(lam)
        self.w, self.h = int(image_size[0]), int(image_size[1])
        if center is None:
            self.cx, self.cy = self.w / 2.0, self.h / 2.0
        else:
            self.cx, self.cy = float(center[0]), float(center[1])
        self.f0 = float(np.hypot(self.w, self.h)) / 2.0

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Distorted pixel coords (N, 2) -> undistorted pixel coords."""
        p = (np.asarray(pts, dtype=float) - [self.cx, self.cy]) / self.f0
        r2 = (p**2).sum(axis=1, keepdims=True)
        return p / (1.0 + self.lam * r2) * self.f0 + [self.cx, self.cy]

    def distort(self, pts: np.ndarray) -> np.ndarray:
        """Undistorted pixel coords (N, 2) -> distorted pixel coords.

        Inverts the division model per point (closed form: solves the
        quadratic ``lam * ru^2 * rd^2 - rd + ru = 0`` for the distorted
        radius rd given the undistorted radius ru, in normalised units).
        """
        if abs(self.lam) < 1e-12:
            return np.asarray(pts, dtype=float).copy()
        p = (np.asarray(pts, dtype=float) - [self.cx, self.cy]) / self.f0
        ru = np.sqrt((p**2).sum(axis=1))
        rd = np.empty_like(ru)
        mask = ru > 1e-12
        a = self.lam * ru[mask] ** 2
        disc = np.maximum(1.0 - 4.0 * a, 0.0)
        # Solve lam * ru * rd^2 - rd + ru = 0 for rd; this root -> ru as lam -> 0.
        rd[mask] = (1.0 - np.sqrt(disc)) / (2.0 * self.lam * ru[mask])
        rd[~mask] = ru[~mask]
        scale = np.ones_like(ru)
        scale[mask] = rd[mask] / ru[mask]
        return p * scale[:, None] * self.f0 + [self.cx, self.cy]
