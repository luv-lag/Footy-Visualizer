# Video pipeline: research notes (Oct 2026)

Background research behind `pipeline/video/`. Numbers come from the cited
papers. Protocols differ between them (detectors, splits, oracle
detections), so treat cross-paper comparisons as rough.

## Player tracking

| Approach | Reported results | Notes |
|---|---|---|
| ByteTrack / OC-SORT / SORT | SportsMOT ~63 HOTA (ByteTrack) | Kalman + IoU. Cheap, CPU-friendly. |
| BoT-SORT | SportsMOT ~69 HOTA | Adds camera-motion compensation and optional ReID. |
| Deep-EIoU (+ GTA post-processing) | SportsMOT 81.0, SoccerNet 85.4 HOTA | Sports-specific. GTATrack (Deep-EIoU + global tracklet association) won SoccerTrack 2025. |
| Sportsort (2025) | SportsMOT 81.3, SoccerNet 88.0 HOTA | Jersey features plus out-of-view re-association. |
| **McByte++ (2026 preprint)** | SportsMOT 79.9 online / **81.5 with GTA**; SoccerNet-tracking 87.5 / **88.6** | Training-free ByteTrack extension. EdgeTAM mask propagation is used only for ambiguous matches; conditional CMC; OSNet re-ID. Runs at 9–19 FPS. Code announced at github.com/tstanczyk95/McBytePlusPlus. Base McByte is in Roboflow `trackers`. |
| MOTIP2 (Oct 2026 preprint) | SportsMOT 76.0, DanceTrack 73.4 HOTA | End-to-end (association as ID prediction). Needs training. |
| SAM2 / SAM 3 + detector | SoccerNet-tracking 51.2 HOTA (RF-DETR + detector-anchored SAM2, ISD 2026) | Masks help occlusion continuity but are slow. SAM 3 soccer work is still mostly practitioner posts. |

**What we do.** Association runs in **pitch space**: foot points are
mapped through the tracked camera and given fixed-size boxes in metres,
then fed to any `trackers` algorithm. On a synthetic pan with perfect
detections, image-space BoT-SORT produced 35–46 IDs for 10 players. Every
pitch-space variant (ByteTrack, McByte, OC-SORT, SORT, BoT-SORT) produced
exactly 10.

**Next upgrades.**
- On the GPU path, McByte++'s mask cue and OSNet re-ID. Their benefit is
  in image space, during occlusions, so they would complement pitch-space
  association rather than replace it.
- Offline GTA-style tracklet association, a richer version of our
  `stitch_tracklets`.

## Camera tracking (homography over time)

- **Anchor + flow + refine** (implemented): pitchcal anchor, LK flow on
  grass pixels with RANSAC, and periodic refinement against the frame's
  markings. On a synthetic pan and zoom: ~0.1–0.25 m error, no
  orientation flips.
- **Anchor accuracy is the limit.** Single-frame pitchcal is
  occasionally ~1 m off on individual frames, and the tracker inherits
  its anchor. Fusing several anchors over a shot is the next accuracy
  step.
- **BroadTrack (WACV 2025)**: a fixed tripod position with only
  pan/tilt/zoom changing. It halves reprojection error versus per-frame
  calibrators. Code: github.com/evs-broadcast/BroadTrack (check the
  license).

## Analysis strategy

- Offline, two passes: a forward pass (detect, follow the camera), then a
  global pass (smooth, track, vote teams, stitch, smooth positions).
- Teams per track: one clustering per clip, then a majority vote per
  track.
- Plausibility gates: off-pitch detections are dropped and speeds over
  12 m/s are flagged. Lost camera means no positions, never a guess.
- Broadcast (phase 2): cut detection, view classification (grass ratio and
  player size), replay detection. Only main-camera shots are analysed.

## Sources

- [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)
- [SoccerNet 2026 Challenges Results](https://arxiv.org/abs/2607.07320) (GSR was not run in 2026)
- [Training-Free Long-Term MOT for Sports Video Analytics (McByte++)](https://arxiv.org/abs/2608.15688)
- [GTA: Global Tracklet Association](https://arxiv.org/abs/2411.08216)
- [GTATrack, SoccerTrack 2025 winner](https://arxiv.org/abs/2602.00484)
- [Sportsort](https://link.springer.com/article/10.1007/s00138-025-01756-y)
- [MOTIP2](https://arxiv.org/abs/2610.10391)
- [Detector-Anchored SAM2 for broadcast soccer MOT](https://aisel.aisnet.org/isd2014/proceedings2026/datascience/25/)
- [Deep HM-SORT](https://arxiv.org/abs/2406.12081)
- [BroadTrack](https://arxiv.org/abs/2412.01721)
- [Roboflow trackers](https://trackers.roboflow.com/latest)
