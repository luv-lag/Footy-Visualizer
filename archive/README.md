# archive/

Everything in this directory was tried, and superseded, on the way to the
pipeline that now lives in `pipeline/`. It is kept for narrative and
portfolio value, not maintained, and **not wired into the live app or test
suite**. Nothing here is guaranteed to run as-is after the reorganization --
several of these directories have their own internal relative-path
assumptions (e.g. sample images, output directories) that pointed at their
original location in the tree and were not rewritten when they were moved
here, on the principle that archived code is frozen for reading, not kept
runnable.

## Two lineages, same shape

This project has two separate "tried, diagnosed, replaced" lineages, and
they follow the exact same pattern: ship something, hit a real failure mode
on real broadcast frames, diagnose it concretely, and build a principled
replacement rather than patching around the symptom.

**Pitch calibration** (three attempts before the current system):
- `video_pipeline/` -- the earliest approach ("observer_recovery"): recovers
  camera pose by tracking players across video keyframes and decomposing the
  resulting homography. No single-frame capability, no explicit
  failure-reporting -- a full separate project structure with its own README
  and requirements.
- `orientation_test/` -- pure-geometry pitch-axis and formation recovery from
  player positions alone, no homography at all.
- `pitch_analysis/` -- pitch-keypoint homography with a convex-hull trust
  layer, plus `synloc_reference.py`, a measurement harness against the
  Spiideo SoccerNet SynLoc ground truth. This is the direct ancestor of
  `eval/synloc/`'s evaluation scripts, and of the PnLCalib-comparison
  methodology now in `eval/eval_comparison/`.
- `approach_4_legacy_tooling/` -- the original standalone CLI/batch harness
  (`pipeline.py`, `batch_run.py`, `player_detection.py`, `visualization.py`,
  `io_utils.py`) built around PnLCalib. Two files from this same approach,
  `calibration.py` and `projection.py`, turned out to be worth keeping and
  now live in `pipeline/pnlcalib_wrapper/` as the production PnLCalib
  fallback -- everything else in that original harness was scaffolding
  around them that the current `frame_analysis.py` orchestration replaced.

What replaced all four: the current pipeline runs `pitchcal` (see
`pipeline/pitchcal/`) as the primary calibrator, with the PnLCalib wrapper
above as a gated fallback when `pitchcal` honestly refuses a frame. See
`docs/M1_overview.md` / `M1_technical.md` / `M1_flowchart.md` for the
narrative writeup of the `pitch_analysis` era specifically (they document
that system's own output, `outputs/pitch_homography/pitch_topdown.png`, not
the current one).

**Team identification** (one attempt before the current system):
- `team_id_v1/` -- `team_separation.py`, plus its two diagnostic tools
  (`team_debug.py`, `run_samples.py`). Assigns players to teams via a
  saturation-weighted hue histogram (color) plus an LBP texture histogram,
  then KMeans.

What replaced it: `pipeline/team_id_v2.py`. Its own module docstring states
the failure modes it was built to fix, and they are a direct match to what
`team_id_v1`'s diagnostic tools (`team_debug.py` especially) were built to
investigate -- white/black/navy kits producing near-zero hue signal,
same-hue-family kits (sky blue vs. royal blue) being inseparable by hue
alone, and pitch-grass bleeding into the color histogram. v2 replaced the
hue histogram with a chromatic/achromatic lightness-binned signature and
dropped the texture feature entirely, rather than patching v1's histogram.

The pattern across both lineages is the same: a diagnostic tool built to
chase a specific, observed failure on real frames, followed by a
replacement whose design directly targets what the diagnosis found -- not
two unrelated instances of "an old version," but one recurring way of
working through this project.

## Other archived material

- `docs/` -- the M1 milestone writeup (see above).
- `outputs/` -- generated images/JSON from the historical approaches above.
  Not committed to git (regeneratable, and not source); present locally only.
- `football-analysis_samples/` -- sample broadcast frames used by the
  historical `football-analysis` subtree specifically (duplicates of
  `data/samples/`, kept alongside the archived code that references them
  by relative path).
