# pitchcal — single-frame broadcast → pitch homography

Maps pixel coordinates in **one static football broadcast frame** to metric
pitch coordinates (105 × 68 m, corner origin) using only the pitch's own
markings, with explicit handling of the pitch-symmetry orientation
ambiguity and per-frame failure reporting. Pure CPU (numpy / OpenCV /
scipy) — zero VRAM, so it runs alongside the detection and team-ID stages.

## Usage

```python
import cv2
from pitchcal import calibrate

frame = cv2.imread("frame.png")
result = calibrate(
    frame,
    player_feet=feet_px,      # optional (N,2): enables formation fallback
    player_labels=team_ids,   # optional (N,): team labels for feet
    player_boxes=boxes_xyxy,  # optional (N,4): masks kits out of detection
)

if result.ok:
    pitch_xy = result.image_to_pitch(points_px)   # (N,2) metres
    px       = result.pitch_to_image(points_m)
    topdown  = result.warp_topdown(frame)         # inspection map
    print(result.confidence, result.fit_px, result.flipped_for_canonical)
else:
    # INSUFFICIENT_FEATURES / NO_CONSENSUS / AMBIGUOUS / LOW_CONFIDENCE / NO_PITCH
    print(result.status, result.diagnostics)
```

A failed result **never** contains a homography — wrong-but-confident
output was the failure mode this design treats as worst. `min_confidence`
(default 0.30) can be raised for a stricter accept/reject trade-off.

## Coordinate convention & the orientation ambiguity

Pitch markings are symmetric under {identity, mirror-x, mirror-y, rot180},
so line/circle geometry alone cannot tell which stadium end is which — a
published problem ("keypoint disambiguation"). pitchcal resolves it in two
steps, both grounded in camera physics rather than template matching:

* **Mirrors are impossible.** For any real camera above the pitch making a
  non-mirrored image, the pitch→image map is orientation-reversing
  (`sign(det H · w³) < 0` at visible pitch points). Solutions containing a
  mirror flip handedness and are rejected outright (`is_proper_view`).
* **Rot180 is resolved by perspective asymmetry.** The touchline nearer
  the camera projects with larger local scale (`|det H| / |w|³`). Output
  is canonicalised so the camera is always on the y < 0 side; the near
  touchline is y = 0. Which physical end is x = 0 remains unknowable from
  a single frame — canonical output is *consistent*, which is what removes
  the near/far-swap failure mode.

## Pipeline

1. **detect.py** — adaptive grass mask → top-hat white-marking mask
   (player boxes optionally masked; blob/squiggle filters drop kit edges)
   → skeleton chains → **lens self-calibration** (one-parameter division
   model fitted by straightening long strips; identity when the frame
   shows no curvature) → ellipse-first conic detection (chords of a large
   projected circle look locally straight, so the circle is claimed
   before line extraction) → straight-line extraction + merging.
2. **solve.py** — three candidate sources, all scored globally:
   * *Strategy A (lines)*: 2-colorings of detected lines into the two
     pencil families (pruned by vanishing-point concurrency), monotone
     assignments to template positions, mixed point/line DLT. Conic
     intersection points (circle × halfway, penalty-arc × penalty-front)
     are injected where applicable. Exactly-degenerate configurations
     (e.g. three concurrent transverse lines + one touchline) are handled
     by **sampling the one-parameter solution family** from the DLT
     nullspace and selecting the member by chamfer score; the x-scaling
     elation left open by conic-on-own-line constraints is closed in
     closed form from the conic's x-extent.
   * *Strategy B (circle)*: the classic broadcast view — centre circle +
     halfway + one longitudinal anchor. Four identified circle points
     (halfway intersections ↔ (cx, cy±r), tangency contacts ↔ (cx±r, cy))
     solve H directly; a tangent-pin iteration + elation correction
     polish it.
   * *Strategy C (players, optional)*: a RANSAC-fitted defensive line is
     treated as parallel to the goal lines, giving the transverse
     vanishing point; with the centre circle + halfway line this solves
     otherwise-underdetermined frames. Probabilistic — only used when
     paint alone fails, flagged `used_player_formation`, confidence
     down-weighted.
3. **Scoring** — trimmed chamfer distance of the projected template
   against detected marking pixels (robust to detection recall), an
   `explained` fraction restricted to pixels the candidate itself places
   on the pitch, a per-structure veto (no major detected line/circle may
   be left stranded), and a view-plausibility check on the mapped frame
   area. Coarse-to-fine robust least-squares refinement.
4. **api.py** — status + confidence assembly, ambiguity detection
   (structurally different runner-up scoring close → `AMBIGUOUS`).

## Validation

Run:

```
.venv/bin/python tests/test_pitchcal.py            # geometric invariants + e2e (11 tests)
.venv/bin/python scripts/validate_synthetic.py 60  # rendered GT suite
.venv/bin/python scripts/validate_synthetic.py 30 --distort
.venv/bin/python scripts/validate_synloc.py 100    # SynLoc val (real GT)
```

Results at time of writing:

* **Synthetic GT suite (60 varied views, both camera sides)**:
  60 % OK; **orientation correct 36/36 (100 %, zero flips)**; median
  position error **0.05 m** on accepted frames; 1/60 wrong-but-OK
  (a sparse corner view at confidence 0.41; raise `min_confidence` to
  0.45 to reject it at the cost of some real-frame recall). Rejected
  frames fail with explicit statuses, dominated by genuinely
  under-determined tight midfield shots.
* **Real broadcast screenshots (5)**: 3 calibrate OK with visually
  verified overlays (fit 0.8–1.4 px); 2 honestly rejected (one hazy
  low-contrast frame, one with the touchline hidden behind boards).
* **Unit tests**: DLT exactness, degeneracy rejection, family sampling,
  handedness/mirror rejection, canonicalisation from both camera sides,
  division-model round-trip, conic tangency identities, formation
  fallback (fails without players, succeeds + flags with them), non-pitch
  rejection.

## Known limitations

* **SynLoc / Spiideo-style rigs**: those frames are virtual-PTZ crops of
  panoramic installs with strong non-centred distortion that a centred
  division model cannot represent; most SynLoc frames are *honestly
  rejected* rather than calibrated. Broadcast-style lenses (mild centred
  barrel) are handled — λ recovery is unit-tested to ~1 % — but the
  heavy-fisheye synthetic suite passes only its clearest frames.
  Supplying an externally-calibrated undistortion before `calibrate()`
  removes this limitation.
* Sparse corner/box views with ≤ 4 visible lines and no arc admit
  self-consistent wrong interpretations; most are caught by the
  ambiguity/consensus checks (and all mirror/flip errors are eliminated
  structurally), but a residual ~2 % slips through at low confidence.
* Non-regulation pitch sizes: pass `PitchModel(length=..., width=...)`.
