"""Unit and integration tests for pitchcal.

Run with:  .venv/bin/python tests/test_pitchcal.py

Covers the geometric invariants the pipeline depends on, and -- most
importantly -- the orientation-disambiguation behaviour: mirror solutions
must be rejected as physically impossible, and 180-degree-rotated
solutions must be canonicalised to the camera-side convention, verified
against rendered ground truth from both sides of the pitch.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

sys.path.insert(0, ".")

from pitchcal.geometry import (  # noqa: E402
    DivisionModel,
    apply_h_w,
    canonicalize,
    dlt_homography,
    dlt_homography_family,
    is_proper_view,
    line_conic_intersections,
    near_side_is_low_y,
    normalize_line,
    tangent_lines_from_point,
)
from pitchcal.pitch_model import PitchModel  # noqa: E402
from pitchcal.synthetic import camera_homography, render_frame  # noqa: E402

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ok   {name}")
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------- #
def gt_H(cam=(50.0, -30.0, 22.0), look=(52.5, 34.0, 0.0), f=1400.0) -> np.ndarray:
    return camera_homography(np.array(cam), np.array(look), f, (1280, 720))


def img_line(H: np.ndarray, P1, P2) -> np.ndarray:
    p1, _ = apply_h_w(H, np.array(P1)[None])
    p2, _ = apply_h_w(H, np.array(P2)[None])
    return normalize_line(
        np.cross([p1[0, 0], p1[0, 1], 1.0], [p2[0, 0], p2[0, 1], 1.0])
    )


def test_dlt_points() -> None:
    H = gt_H()
    Ps = [np.array(p) for p in [(10.0, 10.0), (90.0, 10.0), (90.0, 60.0), (10.0, 60.0)]]
    pp = [(P, apply_h_w(H, P[None])[0][0]) for P in Ps]
    He, _ = dlt_homography(point_pairs=pp)
    t = np.array([[30.0, 20.0], [80.0, 50.0]])
    err = np.abs(apply_h_w(He, t)[0] - apply_h_w(H, t)[0]).max()
    assert err < 1e-6, err


def test_dlt_lines() -> None:
    H = gt_H()
    specs = [
        ((0.0, 68.0), (105.0, 68.0)),
        ((0.0, 13.84), (105.0, 13.84)),
        ((52.5, 0.0), (52.5, 68.0)),
        ((16.5, 13.84), (16.5, 54.16)),
    ]
    lps = [(img_line(H, a, b), (np.array(a), np.array(b))) for a, b in specs]
    He, _ = dlt_homography(line_pairs=lps)
    t = np.array([[30.0, 20.0], [80.0, 50.0]])
    err = np.abs(apply_h_w(He, t)[0] - apply_h_w(H, t)[0]).max()
    assert err < 1e-6, err


def test_dlt_degenerate_rejected() -> None:
    """Concurrent-heavy configurations must not return an arbitrary H."""
    H = gt_H()
    # 3 concurrent transverse lines + nothing else: rank-deficient.
    specs = [
        ((0.0, 0.0), (0.0, 68.0)),
        ((16.5, 0.0), (16.5, 68.0)),
        ((52.5, 0.0), (52.5, 68.0)),
        ((5.5, 0.0), (5.5, 68.0)),
    ]
    lps = [(img_line(H, a, b), (np.array(a), np.array(b))) for a, b in specs]
    He, _ = dlt_homography(line_pairs=lps)
    assert He is None


def test_dlt_family_sampling() -> None:
    """An underdetermined-by-one system yields a family containing truth."""
    H = gt_H()
    specs = [
        ((105.0, 0.0), (105.0, 68.0)),  # goal line
        ((88.5, 13.84), (88.5, 54.16)),  # pen front
        ((99.5, 24.84), (99.5, 43.16)),  # goal front (3 concurrent)
        ((0.0, 0.0), (105.0, 0.0)),  # near touchline
    ]
    lps = [(img_line(H, a, b), (np.array(a), np.array(b))) for a, b in specs]
    t = np.array([[95.0, 20.0], [90.0, 50.0]])
    gt_proj = apply_h_w(H, t)[0]

    def score(He: np.ndarray) -> float:
        return float(np.abs(apply_h_w(He, t)[0] - gt_proj).max())

    fam = dlt_homography_family(line_pairs=lps, score_fn=score)
    assert fam, "no family members"
    best = min(score(He) for He, _ in fam)
    assert best < 1.0, best  # score-guided search recovers the true member


def test_handedness_rejects_mirrors() -> None:
    pitch = PitchModel()
    for cam in [(50.0, -30.0, 22.0), (55.0, 98.0, 25.0)]:
        H = gt_H(cam=cam)
        assert is_proper_view(H, 105, 68, (1280, 720)), cam
        for M in (pitch.mirror_x(), pitch.mirror_y()):
            assert not is_proper_view(H @ M, 105, 68, (1280, 720)), cam
        # rot180 is proper (it is a real camera on the other side) ...
        assert is_proper_view(H @ pitch.rot180(), 105, 68, (1280, 720))


def test_canonicalization() -> None:
    pitch = PitchModel()
    H_south = gt_H(cam=(50.0, -30.0, 22.0))
    H_north = gt_H(cam=(55.0, 98.0, 25.0))
    assert near_side_is_low_y(H_south, 105, 68)
    assert not near_side_is_low_y(H_north, 105, 68)
    Hc, flipped = canonicalize(H_north, 105, 68)
    assert flipped and near_side_is_low_y(Hc, 105, 68)
    Hc2, flipped2 = canonicalize(H_south, 105, 68)
    assert not flipped2
    # Canonicalisation collapses rot180 twins onto one solution.
    Ht, _ = canonicalize(H_south @ pitch.rot180(), 105, 68)
    t = np.array([[30.0, 20.0], [80.0, 50.0]])
    err = np.abs(apply_h_w(Ht, t)[0] - apply_h_w(Hc2, t)[0]).max()
    assert err < 1e-6, err


def test_division_model_roundtrip() -> None:
    for lam in (-0.05, 0.0, 0.12, 0.3):
        dm = DivisionModel(lam, (1280, 720))
        pts = np.random.default_rng(0).uniform([0, 0], [1280, 720], (200, 2))
        rt = dm.distort(dm.undistort(pts))
        assert np.abs(rt - pts).max() < 1e-8, lam


def test_conic_tangents() -> None:
    pitch = PitchModel()
    H = gt_H()
    C_pitch = pitch.circle_conic()
    Hinv = np.linalg.inv(H)
    C_img = Hinv.T @ C_pitch @ Hinv
    C_img /= np.linalg.norm(C_img)
    # Tangents from the longitudinal VP touch at the top/bottom circle pts.
    vp = H @ np.array([1.0, 0.0, 0.0])
    tg = tangent_lines_from_point(vp, C_img)
    assert tg is not None
    _, contacts = tg
    expect = apply_h_w(H, np.array([[52.5, 34 - 9.15], [52.5, 34 + 9.15]]))[0]
    d = min(
        np.abs(contacts - expect).max(),
        np.abs(contacts[::-1] - expect).max(),
    )
    assert d < 1e-4, d
    # Halfway-line intersections hit the same template points.
    l_half = img_line(H, (52.5, 0.0), (52.5, 68.0))
    ipts = line_conic_intersections(l_half, C_img)
    assert ipts is not None
    d2 = min(
        np.abs(ipts - expect).max(),
        np.abs(ipts[::-1] - expect).max(),
    )
    assert d2 < 1e-4, d2


def test_end_to_end_both_sides() -> None:
    """Full pipeline on rendered frames from both camera sides."""
    from pitchcal import calibrate

    for cam, look, f in [
        ((50.0, -30.0, 22.0), (52.5, 34.0, 0.0), 1400.0),
        ((55.0, 98.0, 25.0), (52.5, 34.0, 0.0), 1500.0),
        ((60.0, -30.0, 25.0), (92.0, 34.0, 0.0), 2000.0),
    ]:
        fr = render_frame(np.array(cam), np.array(look), f)
        res = calibrate(fr.image)
        assert res.ok, (cam, res.status)
        pts = np.array([[500.0, 400.0], [700.0, 500.0], [640.0, 300.0]])
        gt = fr.gt_image_to_pitch(pts)
        if cam[1] > 34:
            gt = np.stack([105 - gt[:, 0], 68 - gt[:, 1]], axis=1)
        err = np.linalg.norm(res.image_to_pitch(pts) - gt, axis=1)
        assert err.max() < 3.0, (cam, err)


def test_formation_fallback() -> None:
    """Player-formation VP rescues a frame with only halfway + circle.

    Renders a tight midfield view whose only paint is the halfway line and
    centre circle (both touchlines out of frame), then supplies synthetic
    defensive-line player positions.  Without players this frame must NOT
    calibrate; with them it should, and be flagged as formation-based.
    """
    from pitchcal import calibrate

    cam = np.array([52.0, -38.0, 26.0])
    fr = render_frame(cam, np.array([57.0, 34.0, 0.0]), 4200.0, n_players=0)
    res0 = calibrate(fr.image)
    assert not res0.ok, res0.status  # honest failure without extra info

    # A settled back four on x = 64 (all four project inside the frame),
    # plus scattered others.
    line_pts = np.array([[64.0, 22.0], [64.0, 30.0], [64.0, 38.0], [64.0, 46.0]])
    others = np.array([[58.0, 26.0], [55.0, 40.0], [60.0, 46.0], [50.0, 32.0]])
    all_pts = np.concatenate([line_pts, others])
    feet, w = apply_h_w(fr.H_gt, all_pts)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    inside = (
        (w > 0)
        & (feet[:, 0] > -50)
        & (feet[:, 0] < 1330)
        & (feet[:, 1] > -50)
        & (feet[:, 1] < 770)
    )
    res1 = calibrate(fr.image, player_feet=feet[inside], player_labels=labels[inside])
    assert res1.ok, res1.status
    assert res1.used_player_formation
    pts = np.array([[640.0, 400.0], [500.0, 300.0]])
    gt = fr.gt_image_to_pitch(pts)
    err = np.linalg.norm(res1.image_to_pitch(pts) - gt, axis=1)
    assert err.max() < 4.0, err


def test_no_pitch_frame() -> None:
    from pitchcal import CalibrationStatus, calibrate

    rng = np.random.default_rng(0)
    junk = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    res = calibrate(junk)
    assert not res.ok
    assert res.status in (
        CalibrationStatus.NO_PITCH,
        CalibrationStatus.INSUFFICIENT_FEATURES,
    )


def main() -> None:
    print("pitchcal test-suite")
    check("dlt: point correspondences exact", test_dlt_points)
    check("dlt: line correspondences exact", test_dlt_lines)
    check("dlt: degenerate config rejected", test_dlt_degenerate_rejected)
    check("dlt: 1-dof family sampled, contains truth", test_dlt_family_sampling)
    check("handedness rejects mirror solutions", test_handedness_rejects_mirrors)
    check("camera-side canonicalisation", test_canonicalization)
    check("division model round-trip", test_division_model_roundtrip)
    check("conic tangents + intersections", test_conic_tangents)
    check("end-to-end, both camera sides", test_end_to_end_both_sides)
    check("player-formation fallback", test_formation_fallback)
    check("non-pitch frame fails cleanly", test_no_pitch_frame)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
