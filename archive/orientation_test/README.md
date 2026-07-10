# Football Pitch Orientation Recovery

Empirically recovers the pitch horizontal axis and player formation from a static
football broadcast image — no homography, no pitch model, pure geometry from player
positions.

## Quick start

```bash
pip install -r orientation_test/requirements.txt
export ROBOFLOW_API_KEY=your_key_here
python orientation_test/main.py image.png
python orientation_test/main.py image.png --eps 60 --min-samples 3 --debug
```

Outputs: `orientation_output.png` (always), `orientation_debug.png` (with `--debug`).

---

## CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `image` | — | Path to input broadcast image |
| `--eps` | `80` | DBSCAN neighbourhood radius in pixels |
| `--min-samples` | `4` | Minimum players to form a DBSCAN cluster |
| `--debug` | off | Adds axis overlay to output; writes `orientation_debug.png` |

---

## Pipeline

### Step 1 — Player detection (Roboflow REST API)

The image is base64-encoded and POSTed to the Roboflow hosted inference endpoint
(`football-players-detection-3zvbc/1`). The API key is read from `$ROBOFLOW_API_KEY`.
Only predictions whose `class` is `player` or `goalkeeper` are kept. Roboflow returns
bounding-box centres `(x, y)` in pixel coordinates; these become the raw point cloud.

### Step 2 — First DBSCAN (raw XY)

`sklearn.cluster.DBSCAN` runs on the raw pixel `(x, y)` coordinates with the
user-supplied `eps` and `min-samples`. Points labelled `-1` (noise) are discarded.
The aim is to group spatially close players into candidate formation lines.

### Step 3 — PCA per cluster

For each cluster we compute the principal axis via NumPy SVD:

```
centered = points − centroid
U, sv, Vt = np.linalg.svd(centered, full_matrices=False)
direction  = Vt[0]           # first right singular vector
angle      = degrees(arctan2(direction[1], direction[0]))
aspect_ratio = sv[0] / sv[1] # elongation of the cluster
```

A cluster is a genuine formation line when its player spread along the dominant axis
is much larger than its spread perpendicular to it — captured by `aspect_ratio`.

### Step 4 — Aspect-ratio filter

Only clusters with `aspect_ratio ≥ 2.5` are kept ("passing clusters"). Compact blobs
of players (e.g. a corner-kick scrum) are discarded; elongated rows are retained.

### Step 5 — Weighted average orientation

The pitch horizontal axis is computed as the player-count-weighted circular mean of
the passing-cluster angles:

```
avg_angle = arctan2(Σ n·sin(θ), Σ n·cos(θ))
```

Using the circular mean avoids the ±180° wrapping issue.

### Step 6 — Coordinate rotation

All player positions are rotated by `−avg_angle` so that the recovered pitch
horizontal axis aligns with the image X axis:

```
R = [[cos(−θ), −sin(−θ)],
     [sin(−θ),  cos(−θ)]]
rotated = R @ coords.T
```

In the rotated frame, formation lines become nearly horizontal bands.

### Step 7 — Formation recovery (second DBSCAN on Y)

A second DBSCAN runs on only the Y coordinates of the rotated positions
(`min_samples=1`, same `eps`). Each resulting cluster is a formation row.
Clusters are sorted by their mean Y value (top-to-bottom) and their player
counts are joined with `-` to produce the formation string (e.g. `1-4-3-3`).

### Step 8 — Visualisation (`orientation_output.png`)

| Element | Colour |
|---------|--------|
| Player dots | Per-cluster colour (grey = noise) |
| PCA fitted line per passing cluster | Same per-cluster colour |
| Cluster annotation (angle, AR) | Per-cluster colour |
| Weighted-average pitch-x line through image centre | White |
| Formation legend bar at bottom | White text on black bar |

With `--debug`, a coordinate-axes overlay is added to `orientation_output.png`:

| Element | Colour |
|---------|--------|
| `img-x` / `img-y` axes from top-left | Grey |
| `pitch-x` / `pitch-y` axes from player centroid | Cyan |
| `you` observer arrow (centroid → top of image) | Green |

### Step 9 — Debug schematic (`orientation_debug.png`, requires `--debug`)

A 1100 × 640 canvas with two side-by-side panels:

* **Left — Image Space**: player dots in raw pixel coords with both the image axes
  (grey) and the recovered `pitch-x` at its measured angle (cyan) drawn from the
  centroid.
* **Right — Pitch Space**: player dots after the rotation correction, with `pitch-x`
  now horizontal and `pitch-y` vertical (cyan). The original `img-x` axis is shown
  derotated (grey) to illustrate how much the broadcast camera diverged from a
  true side-on view.

---

## Output to terminal

```
Calling Roboflow API…
Detected players : 22
Clusters found   : 5
Clusters passing aspect-ratio filter (≥ 2.5): 4
  cluster  0  angle=  +8.34°  AR= 6.21  n=4
  cluster  1  angle=  +7.98°  AR= 4.87  n=5
  cluster  2  angle=  +9.10°  AR= 3.44  n=4
  cluster  3  angle=  +8.55°  AR= 5.13  n=5
Weighted avg orientation :  +8.60°
Recovered formation : 1-4-4-2
Saved : orientation_output.png
```

---

## Limitations

* Works best on images where at least two full formation lines are visible and
  players are spatially separated enough for DBSCAN to cluster them.
* The formation string counts every detected player including the goalkeeper; adjust
  `--min-samples` and `--eps` if the count looks wrong.
* Angle ambiguity: `arctan2` returns angles in `(−180°, 180°]`. The pitch-x axis
  has a 180° symmetry (left-to-right and right-to-left are the same line), so the
  sign of the recovered angle indicates which end of the pitch points "right" in the
  image — which depends on the broadcast camera position.
