# Football Tactical Map

One broadcast football frame in &rarr; every visible player's real-world
pitch position out, on a top-down tactical map -- team-colored, with an
honest confidence readout for how much the pipeline trusts its own
calibration on that specific frame.

<!-- TODO: add a demo screenshot or short GIF of the Gradio app here,
     e.g. ![demo](docs/demo.gif) -- captured manually, not generated. -->

## What it does, and why it's hard

A TV broadcast camera isn't looking straight down at the pitch -- it's
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
   and confidence rather than rendering a plausible-looking map when the
   pipeline doesn't actually trust its output.

## Tech stack

- **Detection**: Roboflow-hosted YOLO (`inference-sdk`)
- **Calibration**: OpenCV + numpy/scipy (pitchcal, from-scratch homography
  fitting), PyTorch + HRNet-W48 (PnLCalib, the fallback)
- **Team ID**: scikit-image (LBP/color features), scikit-learn (KMeans)
- **Demo**: Gradio
- **Rendering**: OpenCV, Matplotlib

## Project layout

```
pipeline/     core library -- detection, team ID, calibration cascade, rendering
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

This is a deliberate architectural choice, not disorganization. PnLCalib
needs a specific PyTorch/CUDA build; the rest of the pipeline doesn't need
torch at all. Keeping them in separate venvs means the lightweight path
(detection, team ID, pitchcal, the test suite) never pulls in a multi-GB
CUDA install, and the PnLCalib fallback's dependencies stay isolated from
everything else.

```bash
# 1. project venv (detection, team ID, pitchcal, tests)
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
# (or: pip install -e . / pip install -r requirements.txt, per pyproject.toml)

# 2. PnLCalib venv (the calibration fallback + the demo app, since the app
#    needs the fallback available)
git clone https://github.com/mguti97/PnLCalib vendor/PnLCalib
uv venv vendor/PnLCalib/.venv
uv pip install --python vendor/PnLCalib/.venv/bin/python -r vendor/PnLCalib/requirements.txt
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
.venv/bin/python tests/test_pitchcal.py
.venv/bin/python tests/test_frame_analysis_failures.py
.venv/bin/python tests/test_orientation_roundtrip.py
```

All three run in the lightweight project venv -- the PnLCalib-fallback
tests use a torch-free stub for the calibration half and exercise the
real (numpy-only) projection code, so the torch venv isn't required just
to verify the pipeline's failure handling.

## Challenges / what I learned

**A silent coordinate-frame bug.** PnLCalib's world frame is centre-origin
with +Y pointing the *opposite* way to pitchcal's corner-origin convention.
A naive translation-only conversion between the two silently mirrors every
player position across the pitch's long axis -- and the bug is invisible in
any overlay or top-down visual check, because a football pitch's markings
are symmetric under exactly that flip. It only shows up in actual point
positions, not in anything that looks at the pitch outline. Catching this
took a dedicated handedness test rather than trusting "the overlay looks
right" (see `pipeline/frame_analysis.py`'s `center_to_corner` and
`eval/eval_comparison/validate_orientation_check.py`). It's the kind of bug
this whole codebase ended up organized around being suspicious of: several
of the real failures here were only detectable by specifically checking the
thing that could be silently wrong, not by eyeballing output.

**Comparing two calibration approaches head-to-head, honestly.** Rather than
assuming pitchcal (built for this project) or PnLCalib (an external,
published model) was simply "better," `eval/eval_comparison/` runs both on
the same frames and scores both with the same yardstick, with the explicit
caveat in that scorer's own docstring that it was tuned around pitchcal and
so isn't perfectly neutral -- visual overlay checks are treated as the
primary evidence, not the automated score alone.

**Iterating toward a principled fix, not a patch.** Both major subsystems
here (calibration and team identification) went through multiple
approaches before landing on their current design, and in both cases the
replacement was built to directly target a diagnosed failure mode rather
than to patch symptoms. See `archive/README.md` for that history in more
detail -- it's kept in the repo specifically because the iteration is part
of the story, not something to hide.

## Licensing note

This repository's own code is MIT-licensed (see `LICENSE`). `vendor/PnLCalib`
is a separate, GPLv2-licensed project and is not distributed as part of this
repository -- see Setup above for cloning it independently.
