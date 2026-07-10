"""Failure-path tests for frame_analysis.analyze_frame().

Exercises the cascade branches that the happy-path demos never hit:

  1. Team-separation failure: fewer than 2 usable player crops -> the team
     pass degrades (warning, ``team_id=None`` everywhere, ``team_separation
     is None``) and calibration still runs WITHOUT team labels.
  2. Orientation-gate rejection: PnLCalib "passes" but its homography is a
     physically-impossible mirror (test double) -> the gate must reject it,
     ``calibration_source == "pnlcalib_rejected"``, NO positions -- the
     rejected homography must not be used anyway.
  3. Total calibration failure: pitchcal refuses AND PnLCalib abstains
     (test double) -> ``calibration_source == "failed"``, no positions,
     frame-level data still usable, no crash.

Cases 2/3 replace ``frame_analysis._pnlcalib_modules`` with a torch-free
test double for the calibration side while reusing the REAL production
projection module (``pnlcalib_wrapper.projection`` is numpy-only), so this
file runs in the project venv without PnLCalib's torch environment.

Run:  .venv/bin/python tests/test_frame_analysis_failures.py
"""

from __future__ import annotations

import enum
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import frame_analysis  # noqa: E402
from detection import Detection  # noqa: E402
from frame_analysis import FrameAnalysis, analyze_frame  # noqa: E402

SCR1 = os.path.join(ROOT, "data", "samples", "Screenshot From 2026-06-26 13-45-51.png")
BRASIL = os.path.join(ROOT, "data", "samples", "brasilscotland2.png")
CACHE = os.path.join(ROOT, "data", "detection_cache")
H_PNL_BRASIL = os.path.join(ROOT, "tests", "fixtures", "H_pnlcalib_brasilscotla.npy")

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ok   {name}\n")
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL {name}")
        traceback.print_exc()


def _summary(fa: FrameAnalysis) -> str:
    n_pos = sum(1 for p in fa.players if p.pitch_position_m is not None)
    return (
        f"    source={fa.calibration_source}  players={len(fa.players)} "
        f"with_positions={n_pos}\n"
        f"    team_separation={fa.team_separation}\n"
        f"    detail={fa.calibration_detail}\n"
        f"    warnings={fa.warnings}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────
class _StubGrade(str, enum.Enum):
    PASS = "pass"
    ABSTAIN = "abstain"


@dataclass
class _StubCalibResult:
    grade: _StubGrade
    H: np.ndarray | None
    n_kp: int
    reproj_err: float


class _StubPnlCalibration:
    """Torch-free double for pnlcalib_wrapper.calibration with a scripted result."""

    CalibGrade = _StubGrade

    def __init__(self, result: _StubCalibResult) -> None:
        self._result = result
        self.load_models_calls = 0

    def load_models(self, device: str) -> tuple[None, None]:
        self.load_models_calls += 1
        return None, None

    def calibrate(self, frame_bgr: np.ndarray, model_kp: Any, model_l: Any, device: str) -> _StubCalibResult:
        return self._result


def _patch_pnlcalib(result: _StubCalibResult):
    """Point frame_analysis at the stub calibration + REAL projection."""
    from pnlcalib_wrapper import projection as real_projection  # numpy-only

    stub = _StubPnlCalibration(result)
    frame_analysis._pnlcalib_modules = lambda: (stub, real_projection)
    frame_analysis._PNL_MODELS = None  # reset the model cache between cases
    return stub


def _tiny_detections(w: float = 2384, h: float = 1343) -> list[Detection]:
    """One confident player + one ball: only 1 usable team crop (< 2)."""
    return [
        Detection(
            class_name="player",
            confidence=0.9,
            bbox_center=(w * 0.5, h * 0.6),
            bbox_size=(40.0, 90.0),
            foot_point=(w * 0.5, h * 0.6 + 45.0),
        ),
        Detection(
            class_name="ball",
            confidence=0.8,
            bbox_center=(w * 0.4, h * 0.5),
            bbox_size=(18.0, 18.0),
            foot_point=(w * 0.4, h * 0.5 + 9.0),
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Case 1: team separation failure (fewer than 2 usable players)
# ──────────────────────────────────────────────────────────────────────────
def test_team_separation_failure() -> None:
    orig = frame_analysis.cached_infer
    frame_analysis.cached_infer = lambda image_path, cache_dir: _tiny_detections()
    try:
        fa = analyze_frame(SCR1)
    finally:
        frame_analysis.cached_infer = orig
    print(_summary(fa))
    assert fa.team_separation is None, "team summary must be None when the pass fails"
    assert any("team separation skipped" in w for w in fa.warnings), fa.warnings
    assert all(p.team_id is None and p.team_confidence is None for p in fa.players)
    # Calibration must still run in degraded (label-free) mode: pitchcal
    # does not require team labels (they only feed the formation fallback).
    assert fa.calibration_source == "pitchcal", fa.calibration_source
    assert all(p.pitch_position_m is not None for p in fa.players)


# ──────────────────────────────────────────────────────────────────────────
# Case 2: PnLCalib passes but the orientation gate rejects (mirror double)
# ──────────────────────────────────────────────────────────────────────────
def test_gate_rejects_mirror() -> None:
    H_raw = np.load(H_PNL_BRASIL)  # real PnLCalib output for this frame
    mirror_center = np.diag([1.0, -1.0, 1.0])  # centre-frame mirror across the long axis
    stub = _patch_pnlcalib(
        _StubCalibResult(_StubGrade.PASS, H_raw @ mirror_center, n_kp=17, reproj_err=4.7)
    )
    fa = analyze_frame(BRASIL)  # pitchcal genuinely refuses this frame
    print(_summary(fa))
    assert stub.load_models_calls == 1
    assert fa.calibration_detail["pitchcal_status"] != "ok"
    assert fa.calibration_detail["pnlcalib_orientation_valid"] is False
    assert "mirror" in fa.calibration_detail["pnlcalib_orientation_reason"]
    assert fa.calibration_source == "pnlcalib_rejected", fa.calibration_source
    assert all(p.pitch_position_m is None for p in fa.players), "rejected H must not be used"
    assert len(fa.players) > 0 and fa.image_size[0] > 0  # frame data still usable


# ──────────────────────────────────────────────────────────────────────────
# Case 3: total calibration failure (pitchcal refuses, PnLCalib abstains)
# ──────────────────────────────────────────────────────────────────────────
def test_total_failure() -> None:
    _patch_pnlcalib(
        _StubCalibResult(_StubGrade.ABSTAIN, None, n_kp=2, reproj_err=float("inf"))
    )
    fa = analyze_frame(BRASIL)
    print(_summary(fa))
    assert fa.calibration_source == "failed", fa.calibration_source
    assert fa.calibration_detail["pnlcalib_grade"] == "abstain"
    assert all(p.pitch_position_m is None for p in fa.players)
    assert any("abstained" in w for w in fa.warnings), fa.warnings
    assert len(fa.players) > 0 and fa.team_separation is not None  # frame still usable


if __name__ == "__main__":
    print("frame_analysis failure-path tests")
    check("case 1: team separation failure degrades cleanly", test_team_separation_failure)
    check("case 2: orientation gate rejects mirrored PnLCalib H", test_gate_rejects_mirror)
    check("case 3: total calibration failure -> source='failed'", test_total_failure)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all failure-path tests passed")
