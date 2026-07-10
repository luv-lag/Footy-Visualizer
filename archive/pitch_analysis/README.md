# pitch_analysis

Single broadcast still → players on a top-down pitch (meters) + a confidence
layer that flags which players can be trusted and refuses the rest.

- `football_analysis.py` — player detection (Roboflow YOLOv8) + team separation.
- `pitch_homography.py` — pitch-keypoint homography, convex-hull trust layer, and
  the per-player relative-confidence deliverable.
- `synloc_reference.py` — **measurement harness**: scores the pipeline's
  recovered-geometry accuracy against the Spiideo SoccerNet SynLoc ground truth.
  Pure measurement — it does not modify the pipeline or build features.

## SynLoc reference harness

For each image the harness recovers geometry **from scratch** with
`pitch_homography.py` (keypoints → homography → hull), projects each detected
player's feet through *our* homography, and scores those positions against the
dataset's `position_on_pitch` ground truth (meters).

The per-image calibration SynLoc ships (`camera_matrix` / `undist_poly`) is the
**held-out answer key and is never used for projection**. Scoring runs the
official `sskit.coco.LocSimCOCOeval` in BEV mode (reads our `position_on_pitch`
directly); `position_from_keypoint_index` is asserted unset so the GT
calibration is never consulted.

The pipeline emits *template pixels*; SynLoc uses *meters*. The pitch size agrees
(70×120 m); only a fixed dihedral convention (which goal is +y / mirror)
separates the two frames. That global constant is resolved **once** on a holdout
by ranking all dihedral candidates, and is frozen only when the winner beats the
runner-up decisively (a wrong mirror lands players tens of meters off). The
frozen constant (id, affine, margin) is logged into the results JSON.

### Environment

`xtcocotools` (a hard dependency of the sskit evaluator) has no wheels for the
system Python 3.14, so the harness runs in a dedicated Python 3.11 venv:

```bash
uv venv --python 3.11 .venv-synloc
VIRTUAL_ENV=.venv-synloc uv pip install \
  numpy"<2" scipy opencv-python-headless scikit-learn requests sskit xtcocotools
VIRTUAL_ENV=.venv-synloc uv pip install \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

(`numpy<2` matches the prebuilt `xtcocotools` ABI; `torch`/`torchvision` are
pulled in transitively by `sskit`.)

### Usage

```bash
PY=.venv-synloc/bin/python

# 1) plumbing self-test — no API key, no pipeline (GT + noise through evaluator)
$PY synloc_reference.py --selftest

# 2) real run — needs a Roboflow key; start on a handful of images
ROBOFLOW_API_KEY=... $PY synloc_reference.py --split mini
ROBOFLOW_API_KEY=... $PY synloc_reference.py --split val --limit 25
ROBOFLOW_API_KEY=... $PY synloc_reference.py --split val            # full 6,777

# 3) re-score cached estimates without re-running the pipeline / API
$PY synloc_reference.py --split mini --rescore-only
```

A `ROBOFLOW_API_KEY` is **required** for real runs: the pitch-keypoint model has
no local fallback, so without a key every frame ABSTAINs. Estimates are cached to
`synloc_results/`, so re-scoring is free after the one-time API spend.

### Reported numbers (`synloc_results/synloc_report_*.json`)

- `mAP-LocSim` (official), plus precision / recall / F1 / frame-accuracy at LocSim 0.5.
- Median + IQR position error in meters, split **in-hull (trusted)** vs
  **extrapolated**.
- Error vs distance-from-hull-boundary (binned, meters).
- A bimodality guard that flags a likely wrong global mirror or camera-side split.

> Note: SynLoc images are 4K wide-angle (lens-distorted), not broadcast stills.
> The pipeline fits a planar homography from pitch keypoints, so errors are
> expected to grow away from the fitted hull — which the in-hull/extrapolated
> split is designed to expose.

## Citation

This harness evaluates against the **Spiideo SoccerNet SynLoc** dataset using the
**sskit** devkit (evaluator and camera math):

> Spiideo SoccerNet — SynLoc (Player Localization). sskit devkit:
> https://github.com/Spiideo/sskit

Please also cite SoccerNet per the SoccerNet challenge guidelines when using this
data.
