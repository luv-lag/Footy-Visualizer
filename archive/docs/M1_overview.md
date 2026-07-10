# M1 — Knowing Where Players Are

## What It Does

Given a single frame from a broadcast football match, this tool figures out where
every visible player is standing on the pitch — in real metres, relative to the
centre spot. The output is a flat top-down map of the pitch with each player
placed at their estimated position.

That top-down map is the foundation for anything tactical: how much space a
team is leaving behind its midfield, whether a winger is isolated, how deep the
defensive line is sitting. None of that analysis is possible until you know where
everyone actually is.

**Worked example.** The frame `brasilscotland2.png` (a wide broadcast shot) was
fed to the detection step. The tool found 21 players — 10 on one side, 11 on the
other — and placed each of them on a top-down pitch view. The output image
`outputs/pitch_homography/pitch_topdown.png` shows the result: coloured dots for
each player, overlaid on a scaled pitch template with the centre circle, penalty
areas, and touchlines drawn in.

## Why This Is Hard

The obvious thing to try is to look at where a player appears in the image and
call that their position. But a TV broadcast camera is not looking straight down
at the pitch. It is sitting in a stand or on a crane, tilted at an angle, with a
lens that compresses some parts of the image more than others. A player standing
near the far touchline and one standing on the halfway line can look almost the
same size and appear almost at the same height in the frame, even though one is
30 metres further away than the other.

To get from pixel positions to real positions you have to work out the camera's
viewpoint: where it is, what angle it is pointing at, and how its lens distorts
the image. This is called camera calibration. It is a well-studied problem, but
for broadcast football it is unusually difficult because:

- The camera moves. There is no fixed rig you can calibrate once and reuse.
- The lens is wide-angle and distorts the image, particularly towards the edges.
- Close-up shots may show only a small strip of pitch, leaving you with very
  little geometry to work from.

Several approaches that look promising at first fail for one of these reasons.
Using the apparent height of players to estimate distance fails because lens
distortion makes the height-to-distance relationship nonlinear and different
at every point in the frame. Looking for the point where pitch lines converge in
the distance (the "vanishing point") assumes a perfect pinhole lens — any real
wide-angle lens breaks that assumption.

## How It Was Solved

The key insight is that the pitch has a known size. The dimensions of a football
pitch are fixed by the laws of the game (within a tight range). If you can
figure out how the pitch's marking lines map onto the image, you can work
backwards to figure out where the camera must be sitting and at what angle —
and from there you can place every visible player on the pitch.

The approach uses a pre-trained model called PnLCalib. It was trained on a large
collection of broadcast footage to recognise pitch marking keypoints (corner
flags, penalty spots, where the centre circle meets the halfway line, and so on)
and line segments (touchlines, the penalty arc, the goal area edges). From those
detections it solves for the full camera geometry: not just "where does the pitch
map to in the image" but "where is the camera, how is it rotated, and what is its
focal length." That full solution correctly handles the 3D geometry of a tilted
camera; it does not assume the pitch is being viewed straight on.

Once the camera geometry is known, placing a player is straightforward. The foot
contact point at the bottom of a player's bounding box is their ground position.
That pixel gets mapped backwards through the camera model to the corresponding
point on the ground plane, giving a position in metres from the centre spot.

The only wrinkle is that the model's internal pitch dimensions (a standard FIFA
105×68 m pitch) differ from the synthetic test dataset's pitch (70×120 m). There
is a fixed scale-and-axis conversion between them, verified to be correct by
testing against ground truth.

## How Well It Works

The accuracy was measured against the SpiideoSynLoc dataset: 6,777 synthetic
broadcast frames with camera-calibration ground truth, evaluated on a 50-image
sample.

The tool succeeded in calibrating the camera on **29 of the 50 frames** (58%).
The rest had too few visible pitch markings for the model to lock on. Among the
29 frames it calibrated:

- **5 frames** produced both correct camera geometry and correct pitch orientation
  (telling which end of the pitch is which). On these, median player position
  error was **3.2 metres** across 66 players.

- **16 frames** had correct camera geometry but the wrong pitch orientation —
  the model correctly worked out the camera's viewpoint but mirrored the pitch
  end-for-end. The pitch is nearly symmetrical in appearance, and without an
  external cue (a visible scoreboard, jersey side) the model cannot tell which
  goal is which. On these frames, estimated positions were reflected across the
  halfway line.

- **8 frames** had genuinely wrong camera geometry — the solution found was
  not close to the true viewpoint.

The 3.2-metre accuracy figure is real but comes from a small sample. Whether
it holds across the full dataset remains to be confirmed with the full 6,777-
image evaluation.

## What Comes Next

The orientation problem is the most tractable of the open issues. 72% of
calibrated frames have correct camera geometry — they just need a reliable way
to determine which goal is left and which is right. That is a binary
classification problem and a much simpler one than the full calibration. Solving
it would push the fraction of frames producing usable positions from 17% to 72%
of calibrated frames.

The other open question is completion rate: 42% of frames never calibrate at all because not enough pitch lines are visible. That is a harder problem, since the information simply isn't in the frame — it would require either pooling
information across nearby video frames or falling back to a motion-based approachfor close-up shots.
