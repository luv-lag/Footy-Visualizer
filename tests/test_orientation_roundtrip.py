"""Round-trip verification of the PnLCalib orientation-correction composition.

The fallback path in ``frame_analysis.py`` composes: raw centre-origin H ->
corner frame (parity conversion) -> ``validate_and_canonicalize_homography``
(reject mirrors, correct rot180) -> back to centre frame -> production
``project_players`` -> ``center_to_corner``.  The standalone gate was
validated in isolation (30/30 synthetic twins), but the only real frame to
exercise this exact composition in production (brasilscotland2) needed no
correction -- so "detect a real flip and recover the RIGHT coordinates
through the full round trip" had never been demonstrated.  A sign error in
either frame conversion could survive composition and surface only when a
correction actually fires (the original y-flip bug had the same signature).

Method: take the real, verified-good PnLCalib homography for
brasilscotland2 plus that frame's real cached detections, and drive the
ACTUAL ``analyze_frame`` fallback path (stub only PnLCalib's torch side,
exactly as ``test_frame_analysis_failures`` does) three times:

  baseline  H_orig                          -> accepted, no flip (reference)
  rotated   H_orig @ diag(-1, -1, 1)        -> genuine rot180 in PnLCalib's
            NATIVE centre-origin frame (det=+1, physically valid): must be
            accepted, ``pnlcalib_was_flipped=True``, and -- the decisive
            check -- every player's corner-origin ``pitch_position_m`` must
            match the baseline within centimetres.
  mirror    H_orig @ diag(-1, 1, 1)         -> true reflection (det=-1,
            physically impossible): must be REJECTED, never corrected.

Run:  .venv/bin/python tests/test_orientation_roundtrip.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_frame_analysis_failures import (  # noqa: E402  (shared test doubles)
    _patch_pnlcalib,
    _StubCalibResult,
    _StubGrade,
    check,
    FAILURES,
)

from frame_analysis import FrameAnalysis, analyze_frame  # noqa: E402

BRASIL = os.path.join(ROOT, "Samples", "brasilscotland2.png")
H_PNL_BRASIL = os.path.join(ROOT, "eval_comparison", "out", "H_pnlcalib_brasilscotla.npy")

#: Pass tolerance for the decisive positional check ("within a few cm").
#: The recovery is pure matrix algebra, so an actually-correct composition
#: lands at ~1e-9 m; anything near the tolerance would itself be suspicious.
TOL_M = 0.03

# Centre-origin frame operations, right-composed onto the pitch side of H.
ROT180_CENTER = np.diag([-1.0, -1.0, 1.0])  # (x,y) -> (-x,-y): det=+1, proper rotation
MIRROR_CENTER = np.diag([-1.0, 1.0, 1.0])  # (x,y) -> (-x, y): det=-1, true reflection


def _run_with_h(H_center: np.ndarray) -> FrameAnalysis:
    """Drive the real analyze_frame fallback path with a scripted PnLCalib H.

    Args:
        H_center: centre-origin pitch->image homography to present as
            PnLCalib's output (grade PASS; kp/err copied from the real run).

    Returns:
        The full FrameAnalysis produced by the production code path.
    """
    _patch_pnlcalib(_StubCalibResult(_StubGrade.PASS, H_center, n_kp=17, reproj_err=4.7))
    return analyze_frame(BRASIL)


def _positions(fa: FrameAnalysis) -> dict[int, tuple[float, float] | None]:
    """Map detection_index -> pitch_position_m for every person record."""
    return {p.detection_index: p.pitch_position_m for p in fa.players}


def _compare(
    base: dict[int, tuple[float, float] | None],
    other: dict[int, tuple[float, float] | None],
) -> tuple[float, list[str]]:
    """Compare two position maps player-by-player.

    Args:
        base: reference positions (from the original homography).
        other: positions recovered from the corrected homography.

    Returns:
        (max_abs_error_m, report_lines): the worst per-axis discrepancy in
        metres and a per-player report of (dx, dy) in centimetres.
    """
    assert set(base) == set(other), "player sets differ between runs"
    worst = 0.0
    lines: list[str] = []
    for idx in sorted(base):
        b, o = base[idx], other[idx]
        if (b is None) != (o is None):
            worst = float("inf")
            lines.append(f"    det#{idx:2d}: None-mismatch (base={b}, recovered={o})")
            continue
        if b is None:
            lines.append(f"    det#{idx:2d}: unprojected in both runs (consistent)")
            continue
        dx, dy = o[0] - b[0], o[1] - b[1]
        worst = max(worst, abs(dx), abs(dy))
        lines.append(
            f"    det#{idx:2d}: base=({b[0]:6.2f},{b[1]:6.2f})m  "
            f"recovered=({o[0]:6.2f},{o[1]:6.2f})m  delta=({dx * 100:+.4f},{dy * 100:+.4f})cm"
        )
    return worst, lines


def test_roundtrip_rot180_and_mirror() -> None:
    """The decisive composition check (Part 1 steps 2-6)."""
    H_orig = np.load(H_PNL_BRASIL)

    # Baseline: the real, verified-good homography through the real path.
    fa_base = _run_with_h(H_orig)
    assert fa_base.calibration_source == "pnlcalib_corrected", fa_base.calibration_source
    assert fa_base.calibration_detail["pnlcalib_was_flipped"] is False
    base_pos = _positions(fa_base)
    n_projected = sum(1 for v in base_pos.values() if v is not None)
    print(f"    baseline: {len(base_pos)} people, {n_projected} projected on-pitch")
    assert n_projected >= 10, "baseline must project a meaningful player set"

    # Rotated twin: genuine centre-origin rot180 (det=+1, physically valid).
    fa_rot = _run_with_h(H_orig @ ROT180_CENTER)
    detail = fa_rot.calibration_detail
    print(f"    rotated : valid={detail['pnlcalib_orientation_valid']} "
          f"flipped={detail['pnlcalib_was_flipped']} source={fa_rot.calibration_source}")
    assert detail["pnlcalib_orientation_valid"] is True, "rotation must be ACCEPTED"
    assert detail["pnlcalib_was_flipped"] is True, "the rot180 correction must actually fire"
    assert fa_rot.calibration_source == "pnlcalib_corrected", fa_rot.calibration_source

    # THE DECISIVE CHECK: recovered coordinates must equal the baseline's.
    worst, lines = _compare(base_pos, _positions(fa_rot))
    print("\n".join(lines))
    print(f"    worst per-axis discrepancy: {worst * 100:.6f} cm (tolerance {TOL_M * 100:.0f} cm)")
    assert worst <= TOL_M, (
        f"rot180 recovery landed {worst:.4f} m off the original positions -- "
        "the center->corner->correct->center composition is WRONG; see the "
        "per-player deltas above for magnitude and direction"
    )

    # Mirror twin: true reflection (det=-1) must be rejected, never "fixed".
    fa_mir = _run_with_h(H_orig @ MIRROR_CENTER)
    detail = fa_mir.calibration_detail
    print(f"    mirror  : valid={detail['pnlcalib_orientation_valid']} "
          f"source={fa_mir.calibration_source} reason={detail['pnlcalib_orientation_reason']!r}")
    assert detail["pnlcalib_orientation_valid"] is False, "mirror must be REJECTED"
    assert fa_mir.calibration_source == "pnlcalib_rejected", fa_mir.calibration_source
    assert all(p.pitch_position_m is None for p in fa_mir.players), (
        "a rejected homography must never produce positions"
    )


if __name__ == "__main__":
    print("orientation round-trip composition test (real frame, real detections)")
    check("rot180 recovered exactly + mirror rejected, through the real path",
          test_roundtrip_rot180_and_mirror)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("round-trip composition verified")
