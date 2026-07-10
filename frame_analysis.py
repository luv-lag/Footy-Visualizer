"""frame_analysis.py -- single-frame orchestration: detections -> teams ->
calibration cascade -> metric player positions.

Pipeline (one static broadcast frame in, one :class:`FrameAnalysis` out):

  1. Player detection            (detection.py, Roboflow; JSON-cached)
  2. Team separation             (team_id_v2.py -- BEFORE calibration, so
                                  team labels/boxes can inform pitchcal)
  3. Calibration, pitchcal       (pitchcal.calibrate; refusals are honest)
  4. Calibration fallback        (PnLCalib via approach_4 wrapper; used
                                  only when pitchcal does not return OK)
  5. Foot-point -> pitch metres  (per calibration source, see below)
  6. Assembly                    (FrameAnalysis with per-player records)

Coordinate convention -- READ THIS BEFORE CONSUMING POSITIONS
-------------------------------------------------------------
``PlayerRecord.pitch_position_m`` is ALWAYS corner-origin: (0, 0) to
(105, 68) metres, matching pitchcal's canonical output, regardless of
``calibration_source``:

* ``"pitchcal"``: ``CalibrationResult.image_to_pitch()`` output is stored
  as-is.  Verified corner-origin empirically (synthetic ground-truth check)
  and per pitchcal's docs -- no additional conversion is applied.
* ``"pnlcalib_corrected"``: PnLCalib's production wrapper works in a
  CENTER-origin world frame.  Its raw homography is first passed through
  pitchcal's standalone orientation gate
  (``validate_and_canonicalize_homography``): physically-impossible
  (mirror) solutions are REJECTED (``"pnlcalib_rejected"``, no positions),
  and valid ones are canonicalized (rot180 applied when the pitch ends
  were swapped; ``pnlcalib_was_flipped`` in ``calibration_detail`` records
  whether that fired).  The corrected homography then flows through the
  UNCHANGED production projection path and :func:`center_to_corner` --
  note the Y sign flip there, it is load-bearing.

This module deliberately does NOT modify detection.py, team_id_v2.py,
pitchcal, or PnLCalib/approach_4 -- it only orchestrates them.

Running the demo (needs the PnLCalib venv, which has torch for the
fallback; scikit-image / inference-sdk / python-dotenv must be installed
into it -- environment-only prerequisite):

    ai-workspace/PnLCalib/.venv/bin/python frame_analysis.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from detection import Detection  # noqa: E402
from pitchcal import (  # noqa: E402
    CalibrationResult,
    calibrate as pitchcal_calibrate,
    validate_and_canonicalize_homography,
)
from team_id_v2 import assign_teams_v2, cached_infer  # noqa: E402

# Pitch dimensions shared by both systems' fitted models (FIFA standard).
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0

# Person-like classes that get a PlayerRecord (ball is excluded).
_PERSON_CLASSES: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})

CalibrationSource = Literal[
    "pitchcal", "pnlcalib_corrected", "pnlcalib_rejected", "failed"
]


# ──────────────────────────────────────────────────────────────────────────
# Coordinate conversion (PnLCalib fallback path only)
# ──────────────────────────────────────────────────────────────────────────
def center_to_corner(
    x_center: float,
    y_center: float,
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
) -> tuple[float, float]:
    """Convert PnLCalib centre-origin world metres to corner-origin metres.

    The validated conversion is  **corner (x, y) = (x_c + L/2, W/2 - y_c)**
    (equivalently: centre (x, y) = (x - L/2, W/2 - y)); for a 105 x 68 pitch
    that is ``(x_c + 52.5, 34 - y_c)``.

    THE Y SIGN FLIP IS ESSENTIAL: PnLCalib's world +Y points the OPPOSITE
    way to the corner frame's +Y.  A naive translation-only conversion
    (``x_c + 52.5, y_c + 34``) silently mirrors every position across the
    pitch's long axis -- and the bug is invisible in any overlay/top-down
    visual check, because the pitch marking layout is symmetric under that
    exact flip.  It only shows up in actual point positions (and in the
    handedness test).  This was discovered and fixed in
    ``eval_comparison/run_comparison.py`` (see ``T_CENTER_TO_CORNER`` there
    and the derivation in ``eval_comparison/validate_orientation_check.py``);
    do not "simplify" this function back to a translation.

    Args:
        x_center: PnLCalib world X (length axis, metres, pitch centre = 0).
        y_center: PnLCalib world Y (width axis, metres, pitch centre = 0).
        length: pitch length in metres (default FIFA 105).
        width: pitch width in metres (default FIFA 68).

    Returns:
        (x, y) in the corner-origin frame: x in [0, length], y in [0, width]
        for on-pitch points.
    """
    return x_center + length / 2.0, width / 2.0 - y_center


# Homogeneous point-map matrices for the SAME validated conversion as
# :func:`center_to_corner`, used to change a homography's pitch-coordinate
# frame (right-composition): H_corner = H_center_raw @ _CORNER_TO_CENTER_XY
# and H_center = H_corner @ _CENTER_TO_CORNER_XY.  Both embed the essential
# Y sign flip -- see center_to_corner's docstring; precedent:
# eval_comparison/run_comparison.py (T_CENTER_TO_CORNER).
_CORNER_TO_CENTER_XY: np.ndarray = np.array(
    [[1.0, 0.0, -PITCH_LENGTH_M / 2.0], [0.0, -1.0, PITCH_WIDTH_M / 2.0], [0.0, 0.0, 1.0]]
)
_CENTER_TO_CORNER_XY: np.ndarray = np.array(
    [[1.0, 0.0, PITCH_LENGTH_M / 2.0], [0.0, -1.0, PITCH_WIDTH_M / 2.0], [0.0, 0.0, 1.0]]
)


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PlayerRecord:
    """One detected person with team assignment and metric pitch position.

    Attributes:
        detection_index: index into the original ``detection.infer`` list
            (traceability back to bboxes / debug tooling).
        class_name: "player", "goalkeeper" or "referee".
        confidence: detector confidence in [0, 1].
        foot_point_px: (u, v) pixel of the ground-contact point
            (bbox bottom-centre, as defined by detection.Detection).
        team_id: 0 or 1 (team_id_v2 clustering); ``None`` when unassigned
            (goalkeepers, referees, low-confidence crops, or when the team
            pass could not run for the frame).
        team_confidence: assignment confidence in [0, 1]; ``None`` when
            ``team_id`` is ``None``.
        pitch_position_m: (x, y) in metres, **corner-origin, (0, 0) to
            (105, 68), matching pitchcal's canonical output** -- this
            convention holds for EVERY ``calibration_source`` (PnLCalib's
            centre-origin output is converted via :func:`center_to_corner`
            before being stored here).  ``None`` when the frame has no
            calibration or the point could not be projected.
    """

    detection_index: int
    class_name: str
    confidence: float
    foot_point_px: tuple[float, float]
    team_id: int | None
    team_confidence: float | None
    pitch_position_m: tuple[float, float] | None


@dataclass
class FrameAnalysis:
    """Full single-frame analysis result.

    Attributes:
        image_path: source frame path.
        image_size: (width, height) pixels.
        calibration_source: which system produced the pixel->pitch mapping:
            "pitchcal" (primary); "pnlcalib_corrected" (fallback: PnLCalib
            homography validated + orientation-canonicalized by pitchcal's
            standalone gate before projection); "pnlcalib_rejected"
            (fallback produced a homography but the gate flagged it as
            physically impossible -- it is NOT used, no positions); or
            "failed" (no system produced a usable calibration).
        calibration_detail: per-source diagnostics -- pitchcal status +
            confidence and, when the fallback ran: PnLCalib grade / n_kp /
            reprojection error plus ``pnlcalib_was_flipped`` (did the
            rot180 canonicalization actually fire) and
            ``pnlcalib_orientation_reason``.
        team_separation: summary from team_id_v2 (separation_score,
            low_confidence_frame, excluded_count) or None if the team pass
            could not run.
        players: one :class:`PlayerRecord` per detected person
            (ball detections are excluded).
        warnings: human-readable notes about degraded steps (e.g. team pass
            skipped, fallback unavailable).
        timings_s: wall-clock seconds per pipeline step.
    """

    image_path: str
    image_size: tuple[int, int]
    calibration_source: CalibrationSource
    calibration_detail: dict[str, Any]
    team_separation: dict[str, Any] | None
    players: list[PlayerRecord]
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# PnLCalib fallback (lazy import + cached models: torch env only)
# ──────────────────────────────────────────────────────────────────────────
_PNL_MODELS: tuple[Any, Any] | None = None


def _pnlcalib_modules() -> tuple[Any, Any]:
    """Import approach_4's calibration + projection modules lazily.

    Raises ImportError when the environment cannot run PnLCalib (no torch);
    the caller degrades gracefully.
    """
    fa_root = os.path.join(_HERE, "ai-workspace", "football-analysis")
    if fa_root not in sys.path:
        sys.path.insert(0, fa_root)
    from approach_4 import calibration as pnl_calibration  # noqa: PLC0415
    from approach_4 import projection as pnl_projection  # noqa: PLC0415

    return pnl_calibration, pnl_projection


def _pnlcalib_fallback(
    frame_bgr: np.ndarray,
    person_dets: list[tuple[int, Detection]],
    team_of: dict[int, tuple[int, float]],
    device: str,
) -> tuple[dict[int, tuple[float, float] | None], dict[str, Any]]:
    """Run PnLCalib, gate/canonicalize its homography, project foot points.

    Flow (the gate is a VALIDITY GATE + MATRIX-LEVEL CORRECTOR only; the
    already-validated projection path is reused unchanged):

    1. ``approach_4.calibration.calibrate`` -> raw centre-origin H
       (or ABSTAIN -> empty positions).
    2. Convert H to the corner frame with the VALIDATED parity conversion
       (:data:`_CORNER_TO_CENTER_XY`; note it is NOT a plain translation --
       PnLCalib's +Y is opposite the corner frame's, see
       :func:`center_to_corner`).  pitchcal's ``origin="center"`` option is
       deliberately NOT used here: it assumes a +Y-aligned centre frame
       and would mis-describe PnLCalib's output.
    3. ``validate_and_canonicalize_homography(..., origin="corner")``:
       physically-impossible (mirror) homographies are rejected -- the
       caller must NOT use them; valid ones come back canonicalized
       (rot180 applied if the pitch ends were swapped).
    4. Convert ``corrected_H`` BACK to the centre frame and feed it through
       the untouched production path: ``project_players`` +
       :func:`center_to_corner`.

    Returns:
        (positions, detail): ``positions`` maps detection_index ->
        corner-origin (x, y) metres or None.  ``detail`` carries PnLCalib
        diagnostics plus, when a homography was produced,
        ``pnlcalib_orientation_valid`` / ``pnlcalib_was_flipped`` /
        ``pnlcalib_orientation_reason``.  positions is empty when PnLCalib
        abstained OR the gate rejected the homography (distinguish via
        ``detail``).
    """
    global _PNL_MODELS
    pnl_calibration, pnl_projection = _pnlcalib_modules()
    if _PNL_MODELS is None:
        _PNL_MODELS = pnl_calibration.load_models(device)
    model_kp, model_l = _PNL_MODELS

    res = pnl_calibration.calibrate(frame_bgr, model_kp, model_l, device=device)
    detail: dict[str, Any] = {
        "pnlcalib_grade": res.grade.value,
        "pnlcalib_n_kp": res.n_kp,
        "pnlcalib_reproj_err_px": float(res.reproj_err),
    }
    if res.grade != pnl_calibration.CalibGrade.PASS:
        return {}, detail

    # -- orientation gate (steps 2-4 of the docstring flow) ---------------
    H_corner_in = res.H @ _CORNER_TO_CENTER_XY
    check = validate_and_canonicalize_homography(H_corner_in, frame_bgr)
    detail["pnlcalib_orientation_valid"] = check.is_valid
    detail["pnlcalib_was_flipped"] = check.was_flipped
    detail["pnlcalib_orientation_reason"] = check.reason
    if not check.is_valid:
        # Physically impossible homography: do NOT fall through to using it.
        return {}, detail
    H_center_corrected = check.corrected_H @ _CENTER_TO_CORNER_XY

    # Production projection path: bbox dicts in centre+size format match
    # detection.Detection's bbox_center/bbox_size, and project_players'
    # foot convention (bbox bottom-centre) matches Detection.foot_point.
    players_in = [
        {
            "bbox": {
                "x": det.bbox_center[0],
                "y": det.bbox_center[1],
                "width": det.bbox_size[0],
                "height": det.bbox_size[1],
            },
            "conf": det.confidence,
        }
        for _, det in person_dets
    ]
    labels = [
        (team_of[idx][0] if idx in team_of else -1) for idx, _ in person_dets
    ]  # project_players' own convention: -1 = unknown team
    projected = pnl_projection.project_players(players_in, labels, H_center_corrected)

    positions: dict[int, tuple[float, float] | None] = {}
    for (det_idx, _), proj in zip(person_dets, projected):
        if proj.world_xy is None:
            positions[det_idx] = None
        else:
            # Centre-origin -> corner-origin.  The Y flip in
            # center_to_corner is essential; see its docstring.
            positions[det_idx] = center_to_corner(*proj.world_xy)
    return positions, detail


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────
def analyze_frame(
    image_path: str,
    cache_dir: str = os.path.join(_HERE, "detection_cache"),
    enable_pnlcalib_fallback: bool = True,
    pnl_device: str = "cuda:0",
) -> FrameAnalysis:
    """Analyze one broadcast frame end-to-end.

    Steps: detection -> team separation -> pitchcal calibration ->
    PnLCalib fallback (only if pitchcal is not OK) -> metric positions in
    ONE consistent convention (corner-origin; see module docstring).

    Args:
        image_path: path to the frame image.
        cache_dir: JSON cache directory for Roboflow detections
            (team_id_v2.cached_infer), so reruns don't re-hit the API.
        enable_pnlcalib_fallback: set False to skip the fallback entirely
            (e.g. in an environment without torch).
        pnl_device: torch device string for the PnLCalib models.

    Returns:
        A :class:`FrameAnalysis`.  Never raises for per-step degradation
        (missing teams, failed calibration, unavailable fallback) -- those
        are reported via ``calibration_source``/``warnings``; it does raise
        for a missing/undecodable image or a failed detection call.
    """
    timings: dict[str, float] = {}
    warnings: list[str] = []

    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Image missing or undecodable: {image_path}")
    h, w = frame_bgr.shape[:2]

    # ── 1. detection ────────────────────────────────────────────────────
    t0 = time.time()
    detections = cached_infer(image_path, cache_dir)
    timings["detection"] = round(time.time() - t0, 3)
    person_dets: list[tuple[int, Detection]] = [
        (i, d) for i, d in enumerate(detections) if d.class_name in _PERSON_CLASSES
    ]

    # ── 2. team separation (BEFORE calibration: labels feed pitchcal) ───
    t0 = time.time()
    team_of: dict[int, tuple[int, float]] = {}  # detection_index -> (team, conf)
    team_summary: dict[str, Any] | None = None
    try:
        team_result = assign_teams_v2(image_path, detections)
        team_of = {
            int(idx): (int(a["team_id"]), float(a["confidence"]))
            for idx, a in team_result["assignments"].items()
        }
        team_summary = {
            "separation_score": team_result["separation_score"],
            "low_confidence_frame": team_result["low_confidence_frame"],
            "excluded_count": team_result["excluded_count"],
        }
    except (ValueError, RuntimeError) as exc:
        warnings.append(f"team separation skipped: {exc}")
    timings["team_separation"] = round(time.time() - t0, 3)

    # ── 3. pitchcal calibration (primary) ───────────────────────────────
    t0 = time.time()
    labelled = [(i, d) for i, d in person_dets if i in team_of and d.class_name == "player"]
    player_feet = (
        np.array([d.foot_point for _, d in labelled]) if labelled else None
    )
    player_labels = (
        np.array([team_of[i][0] for i, _ in labelled]) if labelled else None
    )
    player_boxes = (
        np.array(
            [
                [
                    d.bbox_center[0] - d.bbox_size[0] / 2.0,
                    d.bbox_center[1] - d.bbox_size[1] / 2.0,
                    d.bbox_center[0] + d.bbox_size[0] / 2.0,
                    d.bbox_center[1] + d.bbox_size[1] / 2.0,
                ]
                for _, d in person_dets
            ]
        )
        if person_dets
        else None
    )
    pc_result: CalibrationResult = pitchcal_calibrate(
        frame_bgr,
        player_feet=player_feet,
        player_labels=player_labels,
        player_boxes=player_boxes,
    )
    timings["pitchcal"] = round(time.time() - t0, 3)

    calibration_detail: dict[str, Any] = {
        "pitchcal_status": pc_result.status.value,
        "pitchcal_confidence": round(pc_result.confidence, 3),
    }

    positions: dict[int, tuple[float, float] | None] = {}
    source: CalibrationSource = "failed"

    if pc_result.ok:
        # pitchcal's image_to_pitch is already corner-origin canonical
        # (verified empirically + per its docs): store AS-IS, no conversion.
        source = "pitchcal"
        feet = np.array([d.foot_point for _, d in person_dets])
        if len(feet):
            pts = pc_result.image_to_pitch(feet)
            for (det_idx, _), xy in zip(person_dets, pts):
                positions[det_idx] = (float(xy[0]), float(xy[1]))
    elif enable_pnlcalib_fallback:
        # ── 4. PnLCalib fallback, orientation-gated (see _pnlcalib_fallback)
        t0 = time.time()
        try:
            positions, pnl_detail = _pnlcalib_fallback(
                frame_bgr, person_dets, team_of, pnl_device
            )
            calibration_detail |= pnl_detail
            if pnl_detail.get("pnlcalib_orientation_valid") is False:
                source = "pnlcalib_rejected"
                warnings.append(
                    "PnLCalib homography rejected as physically impossible: "
                    + str(pnl_detail.get("pnlcalib_orientation_reason"))
                )
            elif positions:
                source = "pnlcalib_corrected"
            else:
                warnings.append("PnLCalib fallback abstained; no positions")
        except ImportError as exc:
            warnings.append(f"PnLCalib fallback unavailable in this environment: {exc}")
        timings["pnlcalib_fallback"] = round(time.time() - t0, 3)
    else:
        warnings.append("pitchcal did not calibrate and the fallback is disabled")

    # ── 5./6. assemble per-player records ───────────────────────────────
    players = [
        PlayerRecord(
            detection_index=det_idx,
            class_name=det.class_name,
            confidence=det.confidence,
            foot_point_px=(float(det.foot_point[0]), float(det.foot_point[1])),
            team_id=team_of[det_idx][0] if det_idx in team_of else None,
            team_confidence=team_of[det_idx][1] if det_idx in team_of else None,
            pitch_position_m=positions.get(det_idx),
        )
        for det_idx, det in person_dets
    ]

    return FrameAnalysis(
        image_path=image_path,
        image_size=(w, h),
        calibration_source=source,
        calibration_detail=calibration_detail,
        team_separation=team_summary,
        players=players,
        warnings=warnings,
        timings_s=timings,
    )


# ──────────────────────────────────────────────────────────────────────────
# Demo: one frame per calibration_source outcome
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEMO_FRAMES = [
        # pitchcal calibrates this one (verified in the 11-frame evaluation):
        os.path.join(_HERE, "Samples", "Screenshot From 2026-06-26 13-45-51.png"),
        # pitchcal refuses this one; PnLCalib passes -> gated fallback path:
        os.path.join(_HERE, "Samples", "brasilscotland2.png"),
    ]
    for path in DEMO_FRAMES:
        print(f"\n=== {os.path.basename(path)} ===")
        fa = analyze_frame(path)
        print(f"calibration_source : {fa.calibration_source}")
        print(f"calibration_detail : {fa.calibration_detail}")
        print(f"team_separation    : {fa.team_separation}")
        print(f"warnings           : {fa.warnings}")
        print(f"timings_s          : {fa.timings_s}")
        with_pos = [p for p in fa.players if p.pitch_position_m is not None]
        print(f"players            : {len(fa.players)} ({len(with_pos)} with positions)")
        for p in with_pos[:5]:
            x, y = p.pitch_position_m
            team = "--" if p.team_id is None else f"{p.team_id:2d}"
            print(
                f"  det#{p.detection_index:2d} {p.class_name:10} team={team} "
                f"foot_px=({p.foot_point_px[0]:6.1f},{p.foot_point_px[1]:6.1f}) "
                f"pitch_m=({x:6.2f},{y:6.2f})  [corner-origin]"
            )
