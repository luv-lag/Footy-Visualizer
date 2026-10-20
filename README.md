# Football Map

![demo](docs/demo.png)

## Broadcast -> Topdown Map

A TV broadcast camera isn't looking straight down at the pitch. It's
elevated, angled, and lensed such that two players standing the same
distance apart on the pitch can look very different distances apart in
the frame. Recovering real metric positions from a single still image
means recovering the camera's implicit projection (a homography) from
the pitch markings and/or players visible in that one frame, with no
video, no known camera parameters, and no second view to triangulate
against.

The pipeline:

1. **Detection** (`pipeline/detection.py`) -- Roboflow-hosted YOLO model;
   players, goalkeepers, referees, the ball.
2. **Team separation** (`pipeline/team_id_v2.py`) -- classical color-signature
   clustering on jersey crops, not a CNN (crops are often 20-30px wide and
   motion-blurred, where a pretrained embedding buys little).
3. **Pitch calibration** (`pipeline/pitchcal/`) -- the primary calibrator:
   detects pitch markings, fits a homography, and *honestly refuses* frames
   without enough usable geometry rather than guessing.
4. **Calibration fallback** (`pipeline/pnlcalib_wrapper/`) -- when pitchcal
   refuses, a wrapped call into [PnLCalib](https://github.com/mguti97/PnLCalib)
   (a published keypoint-based calibration model), gated through pitchcal's
   own orientation-validity check before its output is trusted.
5. **Assembly** (`pipeline/frame_analysis.py`) -- one consistent, corner-origin
   metric coordinate per player, regardless of which calibrator produced it.
6. **Rendering** (`pipeline/tactical_map.py`) -- the combined view: annotated
   frame, top-down map, and a status strip that states calibration source
   and confidence rather than rendering a plausible looking map when the
   pipeline doesn't actually trust its output.

## Video (work in progress)

`pipeline/video/` extends the single-frame pipeline to clips from one
continuous camera (phase 1; broadcast cut/replay handling comes next):

1. **Camera tracking** (`camera.py`): anchor with pitchcal (optionally the
   PnLCalib fallback), propagate the homography every frame with optical
   flow on grass pixels, refine against the frame's own markings every few
   frames, and report "lost" rather than guess when the fit decays. Frames
   before the first anchor are recovered by propagating backwards.
2. **Tracking in pitch space** (`tracking.py`): foot points are mapped to
   metres before association, so camera pans and zooms don't break IDs
   (image-space IoU trackers fragment badly on small, panning players).
   Any tracker from Roboflow's `trackers` library can be used. Fragments are
   stitched offline by pitch-space reachability.
3. **Teams per track** (`teams.py`): `team_id_v2` signatures are pooled
   across the clip, clustered once, and each track takes a majority vote.
4. **Positions** (`positions.py`): an off-pitch gate (drops stands and
   scoreboard detections), interpolation across the detection stride,
   smoothing, and a speed gate that flags rather than draws implausible jumps.

```bash
.venv/bin/python pipeline/video/run_video.py clip.mp4 --out out/clip \
    --detector hosted --detect-stride 3
```

This writes `tracks.csv` (per frame and track: pitch x/y, team, speed),
`camera.csv`, `minimap.mp4` (annotated frame plus top-down map) and
`summary.json` to `out/clip/`. Detections are cached in
`out/clip/detections.json`, so re-runs don't call the detector again. Use
`--detector local` to run the same model locally through Roboflow's
`inference` package (CPU or GPU).

On multi-sport turf (American-football yard lines over faint or coloured
soccer lines) pitchcal can't read the field and refuses every frame. Pass
`--anchor-file landmarks.json` instead: `frame`, `image_points` (pixels)
and `pitch_points` (the same points in pitch metres) for one frame. The
camera is then carried through the clip from that frame by optical flow.

## Stack

- **Detection**: Roboflow-hosted YOLO (`inference-sdk`)
- **Calibration**: OpenCV + numpy/scipy (pitchcal, from-scratch homography
  fitting), PyTorch + HRNet-W48 (PnLCalib, the fallback)
- **Team ID**: scikit-image (LBP/color features), scikit-learn (KMeans)
- **Demo**: Gradio
- **Rendering**: OpenCV, Matplotlib

## Project layout

```
pipeline/     core library -- detection, team ID, calibration cascade, rendering;
              pipeline/video/ is the video extension
app/          the Gradio demo app
eval/         evaluation and diagnostic tooling (not part of the live app)
tests/        the test suite
archive/      superseded approaches, kept for their narrative value -- see archive/README.md
vendor/       PnLCalib (external, cloned separately -- see Setup)
data/         sample frames + cached detections (tracked); the SoccerNet
              eval dataset (large, gitignored, see Setup)
```

## Setup

### Why two virtual environments

This is a deliberate architectural choice. PnLCalib
needs a specific PyTorch/CUDA build; the rest of the pipeline doesn't need
torch at all. Keeping them in separate venvs means the lightweight path
(detection, team ID, pitchcal, the test suite) never pulls in a multi-GB
CUDA install, and the PnLCalib fallback's dependencies stay isolated from
everything else.

```bash
# 1. project venv (detection, team ID, pitchcal, tests)
uv sync --extra dev
# (or: python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest)

# 2. PnLCalib venv (the calibration fallback + the demo app, since the app
#    needs the fallback available). The demo app (app/app.py) imports
#    straight from pipeline/, so this venv also needs this project's own
#    requirements.txt on top of PnLCalib's -- not just PnLCalib's deps + gradio.
git clone https://github.com/mguti97/PnLCalib vendor/PnLCalib
uv venv vendor/PnLCalib/.venv
uv pip install --python vendor/PnLCalib/.venv/bin/python -r vendor/PnLCalib/requirements.txt
uv pip install --python vendor/PnLCalib/.venv/bin/python -r requirements.txt
uv pip install --python vendor/PnLCalib/.venv/bin/python gradio
# download PnLCalib's SV_kp / SV_lines weights per its own README into
# vendor/PnLCalib/weights/ -- not included here (500MB+, publicly hosted)
```

Roboflow API key (only needed for frames *not* already in
`data/detection_cache/` -- the bundled sample frames and the demo app's
three example frames all have cached detections, so a fresh clone can run
the test suite and the demo without one):

```bash
echo "ROBOFLOW_API_KEY=your_key_here" > .env
```

### Running the demo app

```bash
cd <repo root>
vendor/PnLCalib/.venv/bin/python app/app.py
```

Opens at `http://127.0.0.1:7860`. The three pre-loaded example frames are
chosen to exercise different pipeline paths: pitchcal calibrating directly,
and two frames where pitchcal honestly refuses and the PnLCalib fallback
takes over.

### Running the tests

```bash
.venv/bin/python -m pytest -q
```

The suite runs in the lightweight project venv, and on every push via
GitHub Actions (`.github/workflows/tests.yml`). Each test file can also be
run directly as a script for verbose output. The PnLCalib fallback
tests use a torch-free stub for the calibration half and exercise the
real (numpy only) projection code, so the torch venv isn't required just
to verify the pipeline's failure handling.

## AI 
This project was built with various AI coding tools. AI generated results were verified (the independent evaluation and synthetic test suites throughout `eval/`). 

## Licensing note

This repository's own code is MIT-licensed (see `LICENSE`). `vendor/PnLCalib`
is a separate, GPLv2-licensed project and is not distributed as part of this
repository -- see Setup above for cloning it independently.
