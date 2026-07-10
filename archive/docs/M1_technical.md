# M1 Technical Record — Metric Player Localisation from Broadcast Frames

## Problem

The goal of M1 is to take a single broadcast football frame and recover the
metric (x, y) position of every visible player on the pitch in a consistent,
ground-truth-comparable coordinate frame.

That requires solving camera calibration: recovering the mapping from image
pixels to physical pitch coordinates, given only the frame itself. This is hard
for three compounding reasons specific to broadcast football:

1. **The camera is elevated and angled.** Players are never seen from directly
   above, so you cannot simply read off ground positions from where they appear
   in the image. A player standing at the edge of the pitch and one standing
   at the centre can appear at the same image height depending on where the
   camera is pointed.

2. **The lens is distorted.** Broadcast cameras use wide-angle or telephoto
   lenses; SynLoc images are 4K wide-angle (fisheye-like) frames. A planar
   homography — the standard projective mapping between two planes — does not
   model lens distortion. Applied to a distorted image it produces errors that
   compound with distance from the optical centre.

3. **Pitch lines are not guaranteed in frame.** A close-up shot of a penalty
   box, or a frame showing only one flank, may contain fewer visible pitch
   markings than any landmark-based approach requires.

## Approaches Considered and Rejected

### Simple homography from pitch keypoints

The `pitch_homography.py` module fits a planar homography using detected pitch
line intersections (Roboflow model, 32-vertex SoccerPitchConfiguration). This
works for relative spatial reasoning — the convex-hull trust layer and pairwise
confidence system built on top of it are still useful — but it does not produce
metric positions. It maps to template pixels (cm/10), not metres, and assumes
a pinhole camera. On distorted 4K images the planar model breaks down: the
estimated homography is consistent only inside the convex hull of the inlier
keypoints, and errors grow sharply outside it.

**Rejected for M1 metric accuracy.** Retained as a complementary relative-
geometry layer.

### Player-height depth estimation

Estimating camera geometry from apparent player heights (known ~1.75–1.85 m)
requires inverting the camera projection to recover distance. This is sensitive
to the same lens distortion as homography, since the relationship between image
height and real-world height is only linear for a pinhole camera. Wide-angle
distortion curves that relationship nonlinearly and differently at every image
position.

**Rejected:** same distortion vulnerability as planar homography; also requires
accurate person height estimation, which varies by posture and scale.

### Vanishing-point geometry

Pitch lines (touchlines, half-line, penalty box) converge at vanishing points
that encode camera pose under a pinhole assumption. Standard vanishing-point
methods (`vanishing_point.py` exists in the codebase) can recover elevation
angle and roughly estimate focal length, but again rely on the pinhole model.
On a distorted lens the true vanishing points are displaced and distorted by
the radial profile, making pose recovery unreliable.

**Rejected:** pinhole assumption fails on distorted lenses; also requires
multiple long pitch lines to be visible simultaneously.

### Manual landmark correspondence

Manually labelling a handful of pitch landmarks (centre spot, penalty spots,
corner arcs) per frame and solving the camera pose from those correspondences
can give metric results. The fundamental limitation is the same as homography:
you need reliable landmark detections, and for a restricted view only a few
landmarks are visible. With fewer than 4 non-collinear correspondences the
pose is under-determined or degenerate.

**Rejected:** no robust landmark detector covers sparse-landmark frames; falls
back to degenerate cases too often to be relied on.

## Chosen Approach: PnLCalib

PnLCalib is a pretrained calibration system that jointly detects keypoints and
line segments from pitch appearance using two HRNet-W48 backbone models
(`SV_kp` for keypoints, `SV_lines` for line segments), then runs a heuristic
voting procedure (`FramebyFrameCalib.heuristic_voting`) to recover the full
camera model: intrinsics K, rotation R, and translation t. The result is the
full perspective projection matrix, not a planar homography. This matters because
the full model correctly handles the 3D geometry of the camera–pitch relationship
without assuming a flat-ground pinhole.

The homography H used for ground-plane projection is then derived from (K, R, t):

```
H = K @ [R[:,0], R[:,1], t]
H /= H[2,2]
```

To localise a player, the foot-contact pixel (u, v) is back-projected through
H_inv:

```
[Xp, Yp, w]^T = H_inv @ [u, v, 1]^T
```

giving (Xp/w, Yp/w) in PnLCalib's world frame (FIFA standard pitch, ±52.5 m
length, ±34 m width, origin at centre).

### Coordinate frame conversion

PnLCalib's world frame does not match the SpiideoSynLoc frame. The authoritative
SynLoc frame (established from `sskit/camera.py` `project_on_ground` defaults
and confirmed against annotation ranges) is:

- X axis: width (short axis), ±35 m  (total 70 m)
- Y axis: length (long axis), ±60 m  (total 120 m)
- Origin: pitch centre, metres

The SynLoc pitch is a 70×120 m synthetic pitch; PnLCalib is fitted to a
105×68 m FIFA standard pitch. The conversion is an axis swap plus per-axis
scale:

```
Ys_est =  -Xp / 0.875   # PnLCalib length X → SynLoc length Y, s=52.5/60
Xs_est =  -Yp / 0.9714  # PnLCalib width  Y → SynLoc width  X, s=34/35
```

The sign negation (both axes) was verified empirically by testing all four
sign combinations on 18 well-calibrated frames (intrinsic error < 2 m); the
`--` combination reduced median error to 13.5 m vs 35–79 m for the others.

### Pipeline stages (as implemented)

1. **Player detection** — Roboflow YOLOv8 model (`football-players-detection-3zvbc/1`)
   detects players + goalkeeper. Foot-contact point = bottom-centre of bounding box.
   `approach_4/player_detection.py` wraps this and saves detections as JSON.

2. **PnLCalib inference** — Input frame resized to 540×960 and passed through
   `SV_kp` (keypoint heatmaps) and `SV_lines` (line-segment heatmaps). Peaks
   extracted via max-pool NMS at thresholds KP_THRESHOLD=0.3434 / LINE_THRESHOLD=0.7867.
   `FramebyFrameCalib.heuristic_voting(refine_lines=False)` produces (K, R, t)
   or None when calibration is degenerate (< 4 keypoints surviving threshold).

3. **H recovery** — From (K, R, t), assemble H and invert. This is the mapping
   from ground-plane metres to image pixels; H_inv goes the other way.

4. **Back-projection** — For each player's foot pixel (u, v):
   `(Xp, Yp) = H_inv @ (u, v, 1)` normalised by the third coordinate.

5. **Coordinate conversion** — Apply axis swap and scale to convert from
   PnLCalib world frame to SynLoc metres.

## Accuracy Results

Evaluated against the SpiideoSynLoc validation set, 50-image run
(`synloc_calibration_eval.json`). Ground-truth player positions are from
`annotations/val.json` `position_on_pitch`, which gives (x, y) in the SynLoc
metre frame. Error = Euclidean distance between estimated and GT position.

### Completion

| Outcome | n | % of 50 |
|---|---|---|
| Never calibrated (kp < 4) | 21 | 42% |
| Calibrated | 29 | 58% |

### Failure taxonomy (among the 29 calibrated frames)

| Category | n | % of calibrated | % of total |
|---|---|---|---|
| (a) Good geometry + correct orientation | 5 | 17% | 10% |
| (b) Good geometry + wrong orientation | 16 | 55% | 32% |
| (c) Wrong H (genuine calibration failure) | 8 | 28% | 16% |

Classification criteria:
- "Good geometry" = per-frame intrinsic median error ≤ 5 m. Intrinsic error
  is computed by fitting the optimal affine from GT SynLoc positions to the
  PnLCalib back-projections; residuals measure geometric accuracy independently
  of coordinate convention.
- "Correct orientation" = per-frame position median ≤ 5 m after applying the
  authoritative transform.
- "Wrong orientation" = intrinsic ≤ 5 m but position > 5 m. These frames have
  correct camera geometry but PnLCalib fitted the view for the opposite pitch
  end. The pitch is nearly symmetric (no touchline asymmetry in appearance),
  making this a hard disambiguation for any appearance-based method.

### Position error

All 29 calibrated frames (aggregate, including failures):
- Median: 15.85 m, Mean: 26.1 m, p90: 58.7 m

Category (a) frames only — the 5 frames with correct geometry and correct
orientation (5 frames, 66 players):
- **Median: 3.21 m, Mean: 3.96 m, p90: 8.1 m**

The 3.21 m number is the honest accuracy of PnLCalib on frames where it succeeds.
It comes from 5 frames (000009, 000019, 000028, 000039, 000044) and 66 players.
The sample is small; a larger evaluation is pending.

Intrinsic geometry across all 18 well-calibrated frames (intrinsic < 2 m):
median 1.57 m. This confirms that when PnLCalib finds a valid camera model, the
geometric accuracy of that model is good; the dominant error source is the wrong-
orientation classification failure, not noisy geometry.

## Known Limitations

**Orientation disambiguation.** 55% of calibrated frames have correct camera
geometry but the wrong pitch orientation — PnLCalib's appearance-based vote
cannot distinguish which goal is which on a symmetric view. The identified fix is
a binary orientation classifier or a pitch-context cue (e.g. visible scoreboard
position, asymmetric crowd geometry, team jersey side). Implementing this would
push the usable calibration rate from 17% to 72% of calibrated frames.

**Completion rate.** 42% of frames produce no calibration at all. PnLCalib
requires at least 4 keypoints above threshold (KP_THRESHOLD=0.3434); close-up
shots or frames with minimal pitch markings visible fall below this. Lowering the
threshold risks degenerate H matrices from low-confidence keypoints.

**Small evaluation sample.** The 3.21 m median accuracy figure is from 5 frames
and 66 players. The full SynLoc val set is 6,777 images; the full evaluation
(~45 min on GPU) has not yet been run. These numbers should be treated as
indicative, not definitive.

**SynLoc vs broadcast.** SynLoc images are rendered at 4K from a synthetic
engine with wide-angle optics. PnLCalib's weights were trained on real broadcast
footage. Transfer to SynLoc is imperfect — the non-zero intrinsic error (median
1.57 m even on well-calibrated frames) is partly attributable to this domain
shift, not just the pitch geometry difference.

## Files

| File | Purpose |
|---|---|
| `evaluate_synloc_calibration.py` | 50-image ground-truth eval, produces `synloc_calibration_eval.json` |
| `apply_synloc_transform.py` | Authoritative coordinate-transform diagnostic; confirms sign convention and computes failure taxonomy |
| `diagnose_synloc_failures.py` | Failure classification and radial-distance correlation check |
| `diagnose_synloc_scale.py` | Scale-fitting diagnostic (shows fitted scale is contaminated by orientation flips) |
| `synloc_calibration_eval.json` | 50-image eval results (the source of all numbers in this document) |
| `football-analysis/approach_4/player_detection.py` | Player detection module (Roboflow + KMeans team split) |
| `football-analysis/pitch_analysis/pitch_homography.py` | Roboflow-keypoint planar homography + hull trust (relative geometry only) |
| `PnLCalib/` | PnLCalib repo with `SV_kp` and `SV_lines` weights |
