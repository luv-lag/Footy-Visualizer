# observer_recovery — Football Camera Recovery & Formation Annotation

Recovers camera position (elevation angle, observer distance, focal length) by tracking football players across video keyframes, then uses the recovered perspective transform to run formation detection on every frame and outputs an annotated video.

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.10+ and a [Roboflow](https://app.roboflow.com) API key for the `football-players-detection-3zvbc/1` model.

## Usage

```bash
export ROBOFLOW_API_KEY=your_key_here
python video_pipeline/main.py match.mp4 -o annotated_output.mp4
```

### Arguments

| Argument | Description |
|---|---|
| `input` | Path to input video file |
| `-o / --output` | Output path (default: `output.mp4`) |

## Pipeline

1. **Keyframe extraction** — frames spaced 60 apart across the full video
2. **Player detection** — Roboflow REST API, filtered to `player` and `goalkeeper` classes
3. **Tracking** — nearest-neighbour matching (100px threshold), retains players tracked across ≥4 consecutive keyframes as stable nodes
4. **Homography** — `cv2.findHomography` with RANSAC (`ransacReprojThreshold=3.0`) per consecutive keyframe pair
5. **Focal length estimation** — Hartley-Zisserman SVD method; falls back to `image_width × 1.2` if outside 500–5000px range
6. **Pose decomposition** — `cv2.decomposeHomographyMat`, 4-way ambiguity resolved by positive depth constraint then lowest reprojection error
7. **LM refinement** — `cv2.solvePnPRefineLM`; points with reprojection error >100px excluded
8. **Scale recovery** — visible pitch width mapped to standard 68m; outputs distance (m) and elevation (°) per pair
9. **Weighted averaging** — pairs weighted by inverse mean reprojection error
10. **Formation detection** — DBSCAN on Y coordinates in warped top-down frame, team split by X midpoint
11. **Annotation** — filled circles per player coloured by team, intra-line edges, formation legend bar

## Output

- `output.mp4` — annotated video with formation overlay
- Terminal printout of final camera parameters

## Package Structure

```
main.py        Entry point and argument parser
detection.py   Roboflow API calls and player filtering
tracking.py    Nearest-neighbour keyframe tracking
homography.py  Focal length estimation, homography, decomposition, LM refinement
formation.py   DBSCAN clustering and formation string generation
overlay.py     Drawing and video writing
requirements.txt
README.md
```
