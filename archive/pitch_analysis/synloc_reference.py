#!/usr/bin/env python3
"""
SynLoc Reference Harness — measure pitch_homography.py against ground truth
===========================================================================
Pure *measurement*. Scores the existing pipeline's recovered-geometry accuracy
against the Spiideo SoccerNet SynLoc validation set and reports numbers. It does
NOT modify the pipeline, does NOT enter the challenge, builds NO new features.

What it measures
----------------
For each image we recover camera geometry *from scratch* with the existing
pipeline (pitch keypoints -> homography -> convex hull -> per-player hull-trust),
project each detected player's feet through *our* homography, and score those
positions against the dataset's ``position_on_pitch`` ground truth in meters.

CRITICAL — the provided per-image calibration is the held-out answer key.
SynLoc ships ``camera_matrix`` / ``undist_poly`` per image. We NEVER feed them
into projection. The official evaluator (sskit ``LocSimCOCOeval``) has two modes:

  * calibration mode  — set ``params.position_from_keypoint_index``; it projects
    submitted image keypoints with the GT calibration via ``image_to_ground``.
  * BEV mode          — reads ``det['position_on_pitch']`` directly.

We use BEV mode and assert ``position_from_keypoint_index`` is never set, so the
GT calibration is untouched. We submit positions from our own homography.

Coordinate bridge (the one place GT touches our transform — by design)
----------------------------------------------------------------------
The pipeline emits *template pixels* (cm/10, corner origin, 120x70 m model).
SynLoc uses *meters* (centre origin, 70x120 m). Pitch *size* agrees (sskit's own
``project_on_ground`` defaults are width=70,height=120). Only a fixed dihedral
ambiguity (which goal is +y / mirror) separates the two conventions — a global
*convention* constant, not per-image calibration. We resolve it ONCE on a small
holdout by ranking all dihedral candidates, and refuse to freeze unless the
winner beats the runner-up by a decisive margin (a wrong mirror puts players
tens of meters off, so the correct one wins by a mile when geometry is sound).
The frozen constant (id + affine + margin) is logged into the results JSON.

Scoring reuses sskit; nothing is reimplemented:
  * official  : sskit.coco.LocSimCOCOeval  -> mAP-LocSim, P/R/F1, frame accuracy.
  * per-player: sskit.metric.match (Hungarian) -> matched meter errors, split
                into trusted (in-hull) vs extrapolated, and binned by distance
                from the hull boundary.

Usage
-----
    # plumbing self-test (no API key, no pipeline): GT + noise through the evaluator
    python synloc_reference.py --selftest

    # real run (needs ROBOFLOW_API_KEY): a handful of images first
    ROBOFLOW_API_KEY=... python synloc_reference.py --split mini
    ROBOFLOW_API_KEY=... python synloc_reference.py --split val --limit 25
    ROBOFLOW_API_KEY=... python synloc_reference.py --split val          # full 6,777

    # re-score cached detections without re-running the pipeline / API
    python synloc_reference.py --split mini --rescore-only

Citation
--------
Spiideo SoccerNet SynLoc dataset and the sskit devkit:
  https://github.com/Spiideo/sskit
"""
from __future__ import annotations
import os
import sys
import json
import copy
import argparse
import datetime as _dt
from dataclasses import dataclass, field, asdict

import numpy as np

# ── pipeline (reused UNCHANGED) ────────────────────────────────────────────────
import cv2
from football_analysis import detect_all, DEFAULT_MODEL_ID
from pitch_homography import (
    detect_pitch_keypoints, solve_homography, grade_frame, map_players,
    Grade, TEMPLATE_LEN_PX, TEMPLATE_WID_PX, CONFIG,
)

# ── official evaluator + camera math (reused; never reimplemented) ─────────────
from xtcocotools.coco import COCO
from sskit.coco import LocSimCOCOeval
from sskit.metric import match as sskit_match


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DATA_ROOT = os.path.join(_REPO, "data", "SoccerNet", "SpiideoSynLoc")
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")
DEFAULT_OUT_DIR = os.path.join(_HERE, "synloc_results")

# Pitch half-extents in the SynLoc meter frame. Pitch model = 70 (width) x 120
# (length) m centred at origin; confirmed by GT ranges and sskit project_on_ground
# defaults (width=70, height=120, center=(0,0)).
PITCH_HALF_X = CONFIG.width / 100.0 / 2.0    # 35.0  (short axis, SynLoc x)
PITCH_HALF_Y = CONFIG.length / 100.0 / 2.0   # 60.0  (long axis,  SynLoc y)
TEMPLATE_TO_M = (CONFIG.length / 100.0) / TEMPLATE_LEN_PX  # 0.1 m per template px

# Score operating point for sskit summarize (fixed -> avoids its f1-argmax
# off-by-one and gives a deterministic P/R/F1 readout). mAP-LocSim is threshold
# independent regardless.
SCORE_THRESHOLD = 0.5

# LocSim 0.5 corresponds to ~0.48 m (locsim = exp(ln(0.05)*d^2), tau=1).
LOCSIM_HALF_M = float(np.sqrt(np.log(0.5) / np.log(0.05)))  # ~0.4805 m


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE BRIDGE  (fixed global convention; resolved once, then frozen)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlobalFrame:
    """Frozen template-px -> SynLoc-meters affine + provenance for the JSON."""
    affine: list           # 2x3 row-major, meters = A @ [tx, ty, 1]
    candidate_id: str      # which dihedral candidate was frozen
    winner_median_m: float
    runner_up_median_m: float
    margin_m: float        # runner_up - winner
    decisive: bool
    ranking: list          # [(candidate_id, median_m), ...] ascending

    def apply(self, template_xy: np.ndarray) -> np.ndarray:
        A = np.asarray(self.affine, float).reshape(2, 3)
        pts = np.asarray(template_xy, float).reshape(-1, 2)
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return (hom @ A.T).astype(np.float32)


def _template_corners() -> np.ndarray:
    """Template-px rectangle corners TL, TR, BR, BL (length axis = x)."""
    return np.array([[0.0, 0.0], [TEMPLATE_LEN_PX, 0.0],
                     [TEMPLATE_LEN_PX, TEMPLATE_WID_PX], [0.0, TEMPLATE_WID_PX]],
                    dtype=float)


def _synloc_corners() -> np.ndarray:
    """SynLoc-meter pitch rectangle corners in a fixed cycle (x=±35, y=±60)."""
    return np.array([[-PITCH_HALF_X, -PITCH_HALF_Y], [PITCH_HALF_X, -PITCH_HALF_Y],
                     [PITCH_HALF_X, PITCH_HALF_Y], [-PITCH_HALF_X, PITCH_HALF_Y]],
                    dtype=float)


def _dihedral_candidates() -> dict[str, np.ndarray]:
    """
    All 8 dihedral mappings (4 rotations x 2 reflections) of the template
    corner-cycle onto the SynLoc corner-cycle, each as a 2x3 affine fitted from
    the 4 corner correspondences. The 4 sign variants are the genuine ambiguity;
    the 4 transposed variants have the wrong aspect and exist to lose decisively,
    making the winning margin a visible confidence check.
    """
    tpl = _template_corners()
    syn = _synloc_corners()
    tpl_h = np.concatenate([tpl, np.ones((4, 1))], axis=1)  # 4x3
    cands: dict[str, np.ndarray] = {}
    for refl in (0, 1):
        cyc = syn[::-1] if refl else syn
        for rot in range(4):
            ordered = np.roll(cyc, rot, axis=0)
            # least-squares affine tpl(px,1) -> ordered(m); exact for these 4 pts.
            A, *_ = np.linalg.lstsq(tpl_h, ordered, rcond=None)  # 3x2
            cands[f"r{rot}{'m' if refl else ''}"] = A.T          # 2x3
    return cands


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynLocImage:
    image_id: int
    file_name: str
    path: str
    width: int
    height: int
    gt_xy: np.ndarray              # (G,2) position_on_pitch in meters


def load_synloc(split: str, *, limit: int | None = None) -> tuple[COCO, list[SynLocImage]]:
    ann_path = os.path.join(ANN_DIR, f"{split}.json")
    if not os.path.isfile(ann_path):
        sys.exit(f"[ERROR] annotation file not found: {ann_path}")
    coco = COCO(ann_path)
    img_ids = sorted(coco.getImgIds())
    if limit is not None:
        img_ids = img_ids[:limit]
    images: list[SynLocImage] = []
    for iid in img_ids:
        info = coco.loadImgs(int(iid))[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=int(iid)))
        gt = np.array([a["position_on_pitch"] for a in anns], float).reshape(-1, 2)
        images.append(SynLocImage(
            image_id=int(iid), file_name=info["file_name"],
            path=os.path.join(VAL_IMG_DIR, info["file_name"]),
            width=int(info["width"]), height=int(info["height"]), gt_xy=gt))
    return coco, images


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER  (recover geometry from scratch; calibration never consulted)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlayerEst:
    template_xy: list | None   # [tx, ty] template px, or None when ABSTAIN
    bbox_xywh: list            # COCO [x, y, w, h] from our detector
    score: float               # detector confidence
    hull_trust: float          # [0,1]
    signed_dist_px: float      # <0 inside hull, >0 outside
    in_hull: bool


@dataclass
class ImageEst:
    image_id: int
    file_name: str
    grade: str                 # solid / partial / abstain
    n_players: int
    players: list              # list[PlayerEst]


def run_pipeline(img: SynLocImage, api_key: str) -> ImageEst:
    """Run the existing pipeline on one image; emit template-space estimates."""
    frame = cv2.imread(img.path)
    if frame is None:
        return ImageEst(img.image_id, img.file_name, "abstain", 0, [])

    players, _ball, _backend = detect_all(frame, api_key, DEFAULT_MODEL_ID)
    # Teams are cosmetic for geometry; pass zeros to keep map_players' contract
    # without dragging in team clustering (this harness measures geometry only).
    team_labels = np.zeros(len(players), dtype=int)

    kps = detect_pitch_keypoints(frame, api_key)
    homo = solve_homography(kps)
    grade = grade_frame(homo, frame.shape)
    gps = map_players(players, team_labels, homo, grade)

    ests: list[PlayerEst] = []
    for p, gp in zip(players, gps):
        x1, y1, x2, y2 = p["bbox"]
        ests.append(PlayerEst(
            template_xy=(None if gp.pitch_xy is None else gp.pitch_xy.tolist()),
            bbox_xywh=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            score=float(p.get("conf", 1.0)),
            hull_trust=float(gp.hull_trust),
            signed_dist_px=float(gp.signed_dist_px),
            in_hull=bool(gp.signed_dist_px <= 0.0)))
    return ImageEst(img.image_id, img.file_name, grade.grade.value, len(players), ests)


# ══════════════════════════════════════════════════════════════════════════════
# FRAME FITTING  (rank all candidates; refuse a non-decisive winner)
# ══════════════════════════════════════════════════════════════════════════════

def _median_match_error(est_xy: np.ndarray, gt_xy: np.ndarray) -> tuple[float, int]:
    """Hungarian-assign est->gt (generous threshold) and return (median dist, n)."""
    if len(est_xy) == 0 or len(gt_xy) == 0:
        return float("nan"), 0
    _det, _miss, _extra, dists, _m = sskit_match(gt_xy, est_xy, th=1e9)
    if len(dists) == 0:
        return float("nan"), 0
    return float(np.median(dists)), int(len(dists))


def fit_global_frame(images: list[SynLocImage], ests: list[ImageEst],
                     *, decisive_ratio: float = 2.0, decisive_abs_m: float = 10.0
                     ) -> GlobalFrame:
    """
    Resolve the template<->SynLoc dihedral convention ONCE on a holdout.

    For every candidate affine, transform all estimated template points to
    meters, Hungarian-match to GT per image, and aggregate the median error.
    Pick the minimum; freeze only if it beats the runner-up decisively.
    """
    cands = _dihedral_candidates()
    gt_by_id = {im.image_id: im.gt_xy for im in images}

    scores: dict[str, float] = {}
    for cid, A in cands.items():
        frame = GlobalFrame(A.tolist(), cid, 0, 0, 0, False, [])
        per_img = []
        for e in ests:
            tpl = np.array([p.template_xy for p in e.players if p.template_xy is not None],
                           float).reshape(-1, 2)
            if len(tpl) == 0:
                continue
            est_m = frame.apply(tpl)
            med, n = _median_match_error(est_m, gt_by_id[e.image_id])
            if n > 0 and np.isfinite(med):
                per_img.append(med)
        scores[cid] = float(np.median(per_img)) if per_img else float("inf")

    ranking = sorted(scores.items(), key=lambda kv: kv[1])
    if not np.isfinite(ranking[0][1]):
        sys.exit("[ERROR] frame fit failed: no candidate produced any matched "
                 "players. Geometry was recovered on zero holdout images "
                 "(all ABSTAIN?). Cannot resolve the coordinate convention.")

    win_id, win_m = ranking[0]
    run_m = ranking[1][1] if len(ranking) > 1 else float("inf")
    margin = run_m - win_m
    decisive = (run_m >= decisive_ratio * win_m) or (margin >= decisive_abs_m)

    print("\n  ── global frame candidate ranking (median matched error, m) ──")
    for cid, m in ranking:
        flag = "  <-- WINNER" if cid == win_id else ""
        print(f"     {cid:>4}  {m:8.3f}{flag}")
    print(f"  winner={win_id}  winner_median={win_m:.3f} m  "
          f"runner_up={run_m:.3f} m  margin={margin:.3f} m  decisive={decisive}")

    return GlobalFrame(cands[win_id].tolist(), win_id, win_m, run_m, margin,
                       bool(decisive), [[c, m] for c, m in ranking])


def fit_per_frame_orientation(images: list[SynLocImage], ests: list[ImageEst]
                              ) -> tuple[dict[int, GlobalFrame], list[dict]]:
    """
    GT-ASSISTED diagnostic only: pick, per image, the dihedral candidate that
    minimises that image's median matched error (chosen using ground truth).

    This is NOT a from-scratch result — it grants each frame the orientation the
    pipeline cannot recover on a symmetric pitch. Used solely to isolate
    localization error from orientation failure. Never the headline.
    """
    cands = _dihedral_candidates()
    gt_by = {im.image_id: im.gt_xy for im in images}
    frames: dict[int, GlobalFrame] = {}
    per_image_best: list[dict] = []
    for e in ests:
        tpl = np.array([p.template_xy for p in e.players if p.template_xy is not None],
                       float).reshape(-1, 2)
        if len(tpl) == 0:
            continue
        scored = {}
        for cid, A in cands.items():
            med, n = _median_match_error(
                GlobalFrame(A.tolist(), cid, 0, 0, 0, False, []).apply(tpl),
                gt_by[e.image_id])
            scored[cid] = med if (n > 0 and np.isfinite(med)) else float("inf")
        best = min(scored, key=scored.get)
        frames[e.image_id] = GlobalFrame(cands[best].tolist(), best,
                                         scored[best], 0, 0, False, [])
        per_image_best.append({"image_id": e.image_id, "best": best,
                               "median_m": (None if not np.isfinite(scored[best])
                                            else float(scored[best]))})
    return frames, per_image_best


def global_frame_validity(global_frame: GlobalFrame, per_image_best: list[dict]) -> dict:
    """How well a single global orientation actually fits, per frame."""
    n = len(per_image_best)
    n_consistent = sum(1 for b in per_image_best if b["best"] == global_frame.candidate_id)
    frac = (n_consistent / n) if n else 0.0
    valid = frac >= 0.9 and global_frame.decisive
    if valid:
        verdict = "global frame OK: a single orientation fits nearly all frames."
    else:
        verdict = (f"global frame INVALID: per-frame optimal orientation matches the "
                   f"global winner on only {n_consistent}/{n} frames. The headline "
                   f"mAP-LocSim reflects orientation failure (the pipeline cannot "
                   f"recover a consistent pitch end from symmetric half-views), not "
                   f"pure localization error. See orientation_granted_diagnostic.")
    return {"global_winner": global_frame.candidate_id, "n_images": n,
            "n_consistent_with_global": n_consistent, "frac_consistent": frac,
            "valid": bool(valid), "verdict": verdict,
            "per_image_best": per_image_best}


# ══════════════════════════════════════════════════════════════════════════════
# DETECTIONS  (COCO format + our trust tags; calibration never engaged)
# ══════════════════════════════════════════════════════════════════════════════

def _frame_for(image_id: int, frame: GlobalFrame,
               per_image: dict[int, GlobalFrame] | None) -> GlobalFrame:
    """The single global frame, or this image's frame in per-image mode."""
    return frame if per_image is None else per_image[image_id]


def build_detections(ests: list[ImageEst], frame: GlobalFrame,
                     per_image: dict[int, GlobalFrame] | None = None) -> list[dict]:
    dets: list[dict] = []
    for e in ests:
        fr = _frame_for(e.image_id, frame, per_image)
        for p in e.players:
            if p.template_xy is None:        # ABSTAIN player: no position emitted
                continue
            xy = fr.apply(np.array(p.template_xy, float))[0]
            dets.append({
                "image_id": e.image_id, "category_id": 1,
                "bbox": p.bbox_xywh, "score": p.score,
                "position_on_pitch": [float(xy[0]), float(xy[1])],
                # extra keys — ignored by the evaluator, used for our split:
                "hull_trust": p.hull_trust, "signed_dist_px": p.signed_dist_px,
                "in_hull": p.in_hull})
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — official (LocSimCOCOeval, BEV mode)
# ══════════════════════════════════════════════════════════════════════════════

def score_official(coco_gt: COCO, detections: list[dict]) -> dict:
    if not detections:
        return {"mAP_LocSim": 0.0, "precision_50": 0.0, "recall_50": 0.0,
                "f1_50": 0.0, "frame_accuracy": 0.0, "score_threshold": SCORE_THRESHOLD,
                "note": "no detections (all ABSTAIN / empty)"}
    coco_dt = coco_gt.loadRes(copy.deepcopy(detections))
    ev = LocSimCOCOeval(coco_gt, coco_dt, iouType="bbox")
    # BEV mode: position_from_keypoint_index must NOT exist -> GT calibration
    # (camera_matrix / undist_poly) is never read. Guard it explicitly.
    assert not hasattr(ev.params, "position_from_keypoint_index"), \
        "calibration mode leaked into the reference harness"
    ev.params.score_threshold = SCORE_THRESHOLD
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP_LocSim": float(s[0]),
            "precision_50": float(s[-5]), "recall_50": float(s[-4]),
            "f1_50": float(s[-3]), "score_threshold": float(s[-2]),
            "frame_accuracy": float(s[-1])}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — per-player meter errors (sskit.metric.match), split by trust
# ══════════════════════════════════════════════════════════════════════════════

def per_player_errors(images: list[SynLocImage], ests: list[ImageEst],
                      frame: GlobalFrame,
                      per_image: dict[int, GlobalFrame] | None = None) -> list[dict]:
    """One record per Hungarian-matched (est, gt) pair, tagged with trust."""
    gt_by_id = {im.image_id: im.gt_xy for im in images}
    recs: list[dict] = []
    for e in ests:
        valid = [p for p in e.players if p.template_xy is not None]
        gt = gt_by_id[e.image_id]
        if not valid or len(gt) == 0:
            continue
        fr = _frame_for(e.image_id, frame, per_image)
        est_m = fr.apply(np.array([p.template_xy for p in valid], float))
        _d, _mi, _ex, dists, matches = sskit_match(gt, est_m, th=1e9)
        for (row, col), dist in zip(matches, dists):   # row=est idx, col=gt idx
            p = valid[row]
            recs.append({
                "image_id": e.image_id, "grade": e.grade,
                "error_m": float(dist), "in_hull": p.in_hull,
                "hull_trust": p.hull_trust, "signed_dist_px": p.signed_dist_px,
                "signed_dist_m": p.signed_dist_px * TEMPLATE_TO_M})
    return recs


def _stats(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"n": 0, "median": None, "iqr": None, "mean": None, "p90": None}
    return {"n": int(len(vals)),
            "median": float(np.median(vals)),
            "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
            "mean": float(np.mean(vals)), "p90": float(np.percentile(vals, 90))}


def summarize(recs: list[dict]) -> dict:
    err = np.array([r["error_m"] for r in recs], float)
    in_mask = np.array([r["in_hull"] for r in recs], bool)

    meter_error = {
        "in_hull": _stats(err[in_mask]),
        "extrapolated": _stats(err[~in_mask]),
        "overall": _stats(err),
    }

    # error vs distance from hull boundary (meters; negative = inside)
    sd = np.array([r["signed_dist_m"] for r in recs], float)
    edges = [-np.inf, -20, -10, -5, -2, 0, 2, 5, 10, 20, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sd >= lo) & (sd < hi)
        st = _stats(err[m])
        bins.append({"lo_m": (None if lo == -np.inf else lo),
                     "hi_m": (None if hi == np.inf else hi),
                     "n": st["n"], "median_err_m": st["median"]})

    # bimodality guard: a wrong global mirror shows as a cluster of good players
    # AND a cluster tens of meters off. Flag if both ends are heavily populated.
    if len(err):
        frac_good = float((err < 1.0).mean())
        frac_far = float((err > 15.0).mean())
    else:
        frac_good = frac_far = 0.0
    bimodal = bool(frac_good > 0.25 and frac_far > 0.25)

    return {"meter_error": meter_error,
            "error_vs_hull_distance": bins,
            "bimodality": {"frac_below_1m": frac_good, "frac_above_15m": frac_far,
                           "bimodal_warning": bimodal}}


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(out_dir: str, split: str, limit: int | None) -> str:
    tag = f"{split}" + (f"_n{limit}" if limit else "")
    return os.path.join(out_dir, f"synloc_ests_{tag}.json")


def _report_path(out_dir: str, split: str, limit: int | None) -> str:
    tag = f"{split}" + (f"_n{limit}" if limit else "")
    return os.path.join(out_dir, f"synloc_report_{tag}.json")


def _ests_to_json(ests: list[ImageEst]) -> list:
    return [{"image_id": e.image_id, "file_name": e.file_name, "grade": e.grade,
             "n_players": e.n_players, "players": [asdict(p) for p in e.players]}
            for e in ests]


def _ests_from_json(blob: list) -> list[ImageEst]:
    out = []
    for e in blob:
        players = [PlayerEst(**p) for p in e["players"]]
        out.append(ImageEst(e["image_id"], e["file_name"], e["grade"],
                            e["n_players"], players))
    return out


def evaluate(split: str, *, limit: int | None, api_key: str | None,
             out_dir: str, rescore_only: bool, allow_weak_frame: bool) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    coco_gt, images = load_synloc(split, limit=limit)
    print(f"[load] split={split}  images={len(images)}"
          + (f"  (limit {limit})" if limit else ""))

    cache = _cache_path(out_dir, split, limit)
    if rescore_only:
        if not os.path.isfile(cache):
            sys.exit(f"[ERROR] --rescore-only but no cache at {cache}")
        ests = _ests_from_json(json.load(open(cache))["estimates"])
        print(f"[cache] loaded {len(ests)} image estimates from {cache}")
    else:
        if not api_key:
            sys.exit("[ERROR] ROBOFLOW_API_KEY required for a real run. The pitch-"
                     "keypoint model has no local fallback; without a key every "
                     "frame ABSTAINs. Use --selftest to validate the scoring "
                     "plumbing without a key, or --rescore-only on a cache.")
        ests = []
        for i, im in enumerate(images, 1):
            print(f"[run {i}/{len(images)}] {im.file_name}", flush=True)
            ests.append(run_pipeline(im, api_key))
        json.dump({"split": split, "limit": limit, "estimates": _ests_to_json(ests)},
                  open(cache, "w"))
        print(f"[cache] wrote {cache}")

    n_abstain = sum(1 for e in ests if e.grade == "abstain")
    print(f"[grade] solid={sum(e.grade=='solid' for e in ests)}  "
          f"partial={sum(e.grade=='partial' for e in ests)}  abstain={n_abstain}")

    frame = fit_global_frame(images, ests)
    if not frame.decisive and not allow_weak_frame:
        sys.exit("\n[ERROR] global frame winner is NOT decisive "
                 f"(margin={frame.margin_m:.2f} m). Refusing to freeze a "
                 "convention that didn't win convincingly — add more images or a "
                 "non-degenerate split. Override with --allow-weak-frame.")

    # ── headline: single global frame (honest from-scratch) ──
    detections = build_detections(ests, frame)
    official = score_official(coco_gt, detections)
    recs = per_player_errors(images, ests, frame)
    summ = summarize(recs)

    # ── how valid is a single global orientation, really? ──
    pf_frames, pf_best = fit_per_frame_orientation(images, ests)
    validity = global_frame_validity(frame, pf_best)

    # ── GT-ASSISTED diagnostic: per-frame best orientation (NOT the headline) ──
    diag_dets = build_detections(ests, frame, per_image=pf_frames)
    diag_official = score_official(coco_gt, diag_dets)
    diag_recs = per_player_errors(images, ests, frame, per_image=pf_frames)
    diag_summ = summarize(diag_recs)
    orientation_granted_diagnostic = {
        "gt_assisted": True,
        "warning": ("OPTIMISTIC LOWER BOUND, NOT A FROM-SCRATCH RESULT. Each frame "
                    "is scored under its own error-minimising dihedral flip, chosen "
                    "using ground truth. Use only to separate localization error "
                    "from the pipeline's inability to orient a symmetric pitch."),
        "official": diag_official,
        "meter_error": diag_summ["meter_error"],
        "error_vs_hull_distance": diag_summ["error_vs_hull_distance"],
    }

    report = {
        "meta": {
            "dataset": "SpiideoSynLoc", "split": split, "limit": limit,
            "n_images": len(images), "n_abstain": n_abstain,
            "n_detections": len(detections), "n_matched_players": len(recs),
            "locsim_tau": LocSimCOCOeval.locsim_tau, "locsim_half_m": LOCSIM_HALF_M,
            "score_threshold": SCORE_THRESHOLD,
            "global_frame": asdict(frame),
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "cache": cache,
        },
        "official": official,
        **summ,
        "global_frame_validity": validity,
        "orientation_granted_diagnostic": orientation_granted_diagnostic,
        "per_image": [{"image_id": e.image_id, "file_name": e.file_name,
                       "grade": e.grade, "n_players": e.n_players}
                      for e in ests],
    }
    rpath = _report_path(out_dir, split, limit)
    json.dump(report, open(rpath, "w"), indent=2)
    _print_summary(report, rpath)
    return report


def _print_summary(report: dict, rpath: str) -> None:
    m = report["meta"]; o = report["official"]; me = report["meter_error"]
    print("\n" + "=" * 68)
    print(f"  SynLoc reference — split={m['split']}  images={m['n_images']}  "
          f"abstain={m['n_abstain']}")
    print(f"  frame: {m['global_frame']['candidate_id']}  "
          f"margin={m['global_frame']['margin_m']:.2f} m  "
          f"decisive={m['global_frame']['decisive']}")
    print("-" * 68)
    print(f"  mAP-LocSim            : {o['mAP_LocSim']:.4f}")
    print(f"  Precision/Recall/F1@.5: {o['precision_50']:.3f} / "
          f"{o['recall_50']:.3f} / {o['f1_50']:.3f}")
    print(f"  Frame accuracy        : {o['frame_accuracy']:.3f}")
    print("-" * 68)
    for key in ("in_hull", "extrapolated", "overall"):
        s = me[key]
        if s["n"]:
            iqr = s["iqr"]
            print(f"  err[{key:>12}] n={s['n']:>5}  median={s['median']:.2f} m  "
                  f"IQR=[{iqr[0]:.2f}, {iqr[1]:.2f}]  p90={s['p90']:.2f} m")
        else:
            print(f"  err[{key:>12}] n=0")
    bw = report["bimodality"]
    if bw["bimodal_warning"]:
        print("-" * 68)
        print(f"  ** BIMODAL WARNING ** {bw['frac_below_1m']:.0%} <1 m and "
              f"{bw['frac_above_15m']:.0%} >15 m — possible wrong global mirror "
              "or a camera-side domain split. Inspect before trusting the number.")

    v = report["global_frame_validity"]
    print("-" * 68)
    print(f"  global-frame validity : consistent on "
          f"{v['n_consistent_with_global']}/{v['n_images']} frames "
          f"({v['frac_consistent']:.0%})  valid={v['valid']}")
    if not v["valid"]:
        print(f"  !! {v['verdict']}")

    d = report["orientation_granted_diagnostic"]; dme = d["meter_error"]
    print("-" * 68)
    print("  [GT-ASSISTED DIAGNOSTIC — per-frame best flip, NOT the headline]")
    print(f"  mAP-LocSim (diag)     : {d['official']['mAP_LocSim']:.4f}")
    for key in ("in_hull", "extrapolated", "overall"):
        s = dme[key]
        if s["n"]:
            print(f"  err[{key:>12}] (diag) n={s['n']:>5}  median={s['median']:.2f} m  "
                  f"IQR=[{s['iqr'][0]:.2f}, {s['iqr'][1]:.2f}]")
    print("=" * 68)
    print(f"  report -> {rpath}")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (no API key, no pipeline): GT + noise through the evaluator
# ══════════════════════════════════════════════════════════════════════════════

def selftest(split: str = "mini") -> None:
    coco_gt, images = load_synloc(split)
    rng = np.random.default_rng(42)
    print("  self-test: GT positions + gaussian noise through LocSimCOCOeval (BEV)")
    for noise in (0.0, 0.25, 0.5, 1.0, 3.0):
        dets = []
        for im in images:
            anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=im.image_id))
            for a in anns:
                pos = np.array(a["position_on_pitch"], float)
                if noise:
                    pos = pos + rng.normal(0, noise, size=2)
                dets.append({"image_id": im.image_id, "category_id": 1,
                             "bbox": a["bbox"], "score": 1.0,
                             "position_on_pitch": pos.tolist()})
        res = score_official(coco_gt, dets)
        print(f"  noise={noise:>4} m  ->  mAP-LocSim={res['mAP_LocSim']:.4f}  "
              f"P/R/F1={res['precision_50']:.2f}/{res['recall_50']:.2f}/"
              f"{res['f1_50']:.2f}")
    print("  expected: 1.0 at 0 m, decaying monotonically (LocSim 0.5 ~ 0.48 m).")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Measure pitch_homography.py vs SynLoc GT.")
    ap.add_argument("--split", default="mini",
                    help="annotation split: mini / val / train / test (default mini)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of images (start small!)")
    ap.add_argument("--api-key", default=None, help="Roboflow key (or ROBOFLOW_API_KEY)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--rescore-only", action="store_true",
                    help="re-score cached estimates without re-running the pipeline")
    ap.add_argument("--allow-weak-frame", action="store_true",
                    help="proceed even if the global-frame winner is not decisive")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the evaluator plumbing with GT+noise (no key)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.split)
        return

    key = (args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")).strip() or None
    evaluate(args.split, limit=args.limit, api_key=key, out_dir=args.out_dir,
             rescore_only=args.rescore_only, allow_weak_frame=args.allow_weak_frame)


if __name__ == "__main__":
    main()
