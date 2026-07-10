"""Team-differentiation stage, v2 -- chromatic/achromatic colour signatures.

An alternative to team_separation.py, built around the failure modes observed
on the sample set:

1. WHITE / BLACK / NAVY KITS: a saturation-weighted hue histogram (v1) throws
   away exactly the pixels that identify an achromatic kit -- a white shirt
   has no stable hue at all, so v1 sees near-zero signal and clusters on
   noise. v2 routes low-chroma pixels into dedicated *lightness* bins
   (black / dark / light / white) instead of discarding them, so "white team
   vs blue team" becomes a trivially separable signature.

2. SAME-HUE-FAMILY KITS (sky blue vs royal blue): hue alone can't split
   them. v2 splits every hue bin into a dark and a light band, so lightness
   differences within one hue family remain visible in the signature.

3. TWO-TONE KITS (e.g. red/white halves): any "single dominant colour"
   descriptor flips between the two tones from player to player and splits
   the team. A *distribution* signature keeps both tones present in every
   player's vector, so intra-team distances stay small.

4. GRASS BLEED-THROUGH: pitch pixels inside the torso box are suppressed by
   an excess-green test (2G - R - B), which is white-balance-robust and needs
   no per-frame estimation, before any binning happens.

The final per-player signature is an L1-normalized histogram mapped through a
square root (Hellinger mapping), so Euclidean distance between signatures
approximates the Hellinger distance between colour distributions -- a proper
metric for histograms -- and plain KMeans stays statistically meaningful.

Output contract mirrors team_separation.assign_teams: assignments keyed by
index into the ORIGINAL detections list, a frame-level separation score, and
a low-confidence flag. Nothing is withheld on low confidence.

Run as a script to batch the whole sample set with cached detections:

    python team_id_v2.py [samples_dir]
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from detection import Detection, infer

# --------------------------------------------------------------------------
# Torso cropping
# --------------------------------------------------------------------------

#: Fractions of the player bbox trimmed when isolating the jersey torso.
#: Slightly tighter than v1: the top cut clears head/neck, the bottom cut
#: clears shorts/socks, and the *horizontal* shrink (new in v2) drops the
#: outer margins of the box, which are mostly background/arms/opponents.
TORSO_TOP_FRACTION = 0.16
TORSO_BOTTOM_FRACTION = 0.42
TORSO_HORIZONTAL_KEEP = 0.70

#: Players below this detection confidence are excluded from clustering.
#: Lower than v1's 0.6: the signature is robust enough that even marginal
#: far-sideline crops carry usable colour, and excluding fewer players means
#: fewer "unassigned" boxes downstream.
MIN_DETECTION_CONFIDENCE = 0.30

# --------------------------------------------------------------------------
# Grass suppression
# --------------------------------------------------------------------------

#: Excess-green threshold (on 0-255 channel scale): a pixel with
#: 2G - R - B above this is treated as pitch grass. Chosen high enough that
#: white kit (R=G=B, ExG=0) and even fairly green-tinted floodlit kit stay
#: untouched, while turf -- whose green channel dominates strongly -- is
#: caught reliably across day/night frames without per-frame estimation.
GRASS_EXG_THRESHOLD = 28.0

#: Residual weight multiplier for grass-classified pixels (soft, not hard,
#: suppression -- same rationale as v1's GRASS_SUPPRESSION_FACTOR).
GRASS_RESIDUAL_WEIGHT = 0.02

# --------------------------------------------------------------------------
# Signature binning
# --------------------------------------------------------------------------

#: Number of hue bins for chromatic pixels. 12 bins of 30 real-world degrees
#: each: coarse enough to be blur-tolerant, fine enough that sky blue
#: (~200 deg) and royal blue (~230 deg) land in different bins.
SIGNATURE_HUE_BINS = 12

#: Soft chromaticity gate: a pixel's "chromatic weight" ramps from 0 to 1 as
#: saturation crosses [CHROMA_SAT_LOW, CHROMA_SAT_HIGH] (and similarly for
#: value across [CHROMA_VAL_LOW, CHROMA_VAL_HIGH]); the remainder of the
#: pixel's weight is routed to the achromatic lightness bins. A soft ramp
#: avoids the knife-edge instability of a hard saturation threshold on
#: washed-out broadcast colour.
CHROMA_SAT_LOW = 0.18
CHROMA_SAT_HIGH = 0.30
CHROMA_VAL_LOW = 0.12
CHROMA_VAL_HIGH = 0.22

#: Each chromatic hue bin is split into a dark and a light band; the split is
#: a soft linear ramp over value in [VALUE_BAND_LOW, VALUE_BAND_HIGH]. This
#: is what keeps royal blue (dark band) apart from sky blue (light band).
VALUE_BAND_LOW = 0.35
VALUE_BAND_HIGH = 0.65

#: Number of lightness bins for achromatic pixels (black -> white).
ACHROMATIC_BINS = 4

#: Total signature length.
SIGNATURE_LENGTH = SIGNATURE_HUE_BINS * 2 + ACHROMATIC_BINS

#: Spatial weighting: Gaussian falloff (in normalized [-1, 1] coordinates)
#: emphasizing the middle of the torso crop, where jersey pixels dominate,
#: over the edges, where background and neighbouring players leak in.
#: Horizontal falloff is stronger than vertical because the torso is tall.
SPATIAL_FALLOFF_X = 2.0
SPATIAL_FALLOFF_Y = 0.5

# --------------------------------------------------------------------------
# Clustering / confidence
# --------------------------------------------------------------------------

#: Below this silhouette score the frame's split is flagged low-confidence
#: (same semantics as v1: assignments still returned, just not trusted).
MIN_SILHOUETTE_SCORE = 0.20


def extract_torso_crop(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Crop the jersey torso from a player detection.

    Trims head/neck from the top, shorts/legs from the bottom, and keeps only
    the central TORSO_HORIZONTAL_KEEP fraction of the width.

    Args:
        image: Full BGR frame.
        detection: Player detection to crop.

    Returns:
        BGR crop (a view into `image`).

    Raises:
        ValueError: If the crop would be empty after clamping.
    """
    height, width = image.shape[:2]
    cx, cy = detection.bbox_center
    bw, bh = detection.bbox_size

    keep_half_width = (bw * TORSO_HORIZONTAL_KEEP) / 2.0
    x1 = max(0, min(int(round(cx - keep_half_width)), width))
    x2 = max(0, min(int(round(cx + keep_half_width)), width))

    top = cy - bh / 2.0
    y1 = max(0, min(int(round(top + TORSO_TOP_FRACTION * bh)), height))
    y2 = max(0, min(int(round(top + (1.0 - TORSO_BOTTOM_FRACTION) * bh)), height))

    if y2 <= y1 or x2 <= x1:
        raise ValueError(
            f"Degenerate torso crop for bbox_center={detection.bbox_center}, "
            f"bbox_size={detection.bbox_size} in {width}x{height} frame"
        )
    return image[y1:y2, x1:x2]


def compute_signature(crop: np.ndarray) -> np.ndarray:
    """Build the chromatic/achromatic colour signature for one torso crop.

    Pipeline per pixel: spatial centre-weighting -> grass suppression via
    excess-green -> soft split of the pixel's weight into a chromatic part
    (soft-binned by hue x {dark, light} value band) and an achromatic part
    (soft-binned by lightness only). The result is L1-normalized then
    square-rooted (Hellinger mapping) so downstream Euclidean distances
    approximate a proper histogram metric.

    Args:
        crop: BGR torso crop.

    Returns:
        Signature vector of length SIGNATURE_LENGTH (all-zero if the crop
        carries no usable weight, e.g. pure grass).
    """
    if crop.size == 0:
        return np.zeros(SIGNATURE_LENGTH)

    crop_f = crop.astype(np.float64)
    blue, green, red = crop_f[:, :, 0], crop_f[:, :, 1], crop_f[:, :, 2]

    h_px, w_px = crop.shape[:2]
    xs = np.linspace(-1.0, 1.0, w_px) if w_px > 1 else np.zeros(1)
    ys = np.linspace(-1.0, 1.0, h_px) if h_px > 1 else np.zeros(1)
    weight = np.outer(
        np.exp(-SPATIAL_FALLOFF_Y * ys**2), np.exp(-SPATIAL_FALLOFF_X * xs**2)
    )

    excess_green = 2.0 * green - red - blue
    weight = np.where(excess_green > GRASS_EXG_THRESHOLD, weight * GRASS_RESIDUAL_WEIGHT, weight)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float64)  # [0, 180)
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    val = hsv[:, :, 2].astype(np.float64) / 255.0

    sat_ramp = np.clip((sat - CHROMA_SAT_LOW) / (CHROMA_SAT_HIGH - CHROMA_SAT_LOW), 0.0, 1.0)
    val_ramp = np.clip((val - CHROMA_VAL_LOW) / (CHROMA_VAL_HIGH - CHROMA_VAL_LOW), 0.0, 1.0)
    chromatic_fraction = sat_ramp * val_ramp

    chrom_weight = (weight * chromatic_fraction).ravel()
    achro_weight = (weight * (1.0 - chromatic_fraction)).ravel()

    signature = np.zeros(SIGNATURE_LENGTH)

    # -- chromatic part: soft circular hue binning x soft dark/light band --
    hue_position = (hue.ravel() / 180.0) * SIGNATURE_HUE_BINS - 0.5
    lower_bin = np.floor(hue_position)
    frac = hue_position - lower_bin
    bin_a = lower_bin.astype(int) % SIGNATURE_HUE_BINS
    bin_b = (bin_a + 1) % SIGNATURE_HUE_BINS

    light_band = np.clip(
        (val.ravel() - VALUE_BAND_LOW) / (VALUE_BAND_HIGH - VALUE_BAND_LOW), 0.0, 1.0
    )
    # chromatic layout: index = hue_bin * 2 + band  (band 0 = dark, 1 = light)
    for hue_bin, hue_share in ((bin_a, 1.0 - frac), (bin_b, frac)):
        base = chrom_weight * hue_share
        np.add.at(signature, hue_bin * 2, base * (1.0 - light_band))
        np.add.at(signature, hue_bin * 2 + 1, base * light_band)

    # -- achromatic part: soft lightness binning --
    achro_position = np.clip(val.ravel() * ACHROMATIC_BINS - 0.5, 0.0, ACHROMATIC_BINS - 1.0)
    lower_bin = np.floor(achro_position)
    frac = achro_position - lower_bin
    bin_a = lower_bin.astype(int)
    bin_b = np.minimum(bin_a + 1, ACHROMATIC_BINS - 1)
    offset = SIGNATURE_HUE_BINS * 2
    np.add.at(signature, offset + bin_a, achro_weight * (1.0 - frac))
    np.add.at(signature, offset + bin_b, achro_weight * frac)

    total = signature.sum()
    if total <= 0:
        return np.zeros(SIGNATURE_LENGTH)
    return np.sqrt(signature / total)


def cluster_signatures(signatures: np.ndarray) -> tuple[list[int], list[float], float]:
    """Split Hellinger-mapped signatures into two teams with KMeans.

    Args:
        signatures: Array (n_players, SIGNATURE_LENGTH).

    Returns:
        (team_ids, confidences, separation_score):
        - team_ids: 0/1 per player, in input order.
        - confidences: per-player margin in [0, 1] --
          (d_other - d_assigned) / (d_other + d_assigned); 1.0 means the
          player sits on its centroid, ~0 means it is equidistant from both.
        - separation_score: silhouette score of the split (-1.0 in the
          degenerate single-cluster case, which trips the low-confidence
          flag downstream).

    Raises:
        ValueError: If fewer than 2 signatures are given.
    """
    n_samples = signatures.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 players to split into teams, got {n_samples}")

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=25)
    labels = kmeans.fit_predict(signatures)

    distances = np.linalg.norm(
        signatures[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2
    )
    d_assigned = distances[np.arange(n_samples), labels]
    d_other = distances[np.arange(n_samples), 1 - labels]
    denominator = d_assigned + d_other
    confidences = np.where(
        denominator > 0, (d_other - d_assigned) / denominator, 0.0
    ).tolist()

    if len(set(labels.tolist())) < 2:
        separation = -1.0
    else:
        separation = float(silhouette_score(signatures, labels))

    return labels.tolist(), confidences, separation


def assign_teams_v2(image_path: str, detections: list[Detection]) -> dict:
    """Split detected players into two teams for one broadcast frame.

    Args:
        image_path: Path to the frame the detections came from.
        detections: FULL detection list from infer() (all classes).

    Returns:
        {
            "assignments": {<original detection index>: {"team_id": 0|1,
                                                          "confidence": float}},
            "separation_score": float,
            "low_confidence_frame": bool,
            "excluded_count": int,   # players dropped by confidence/crop filters
        }
        Same contract as team_separation.assign_teams: keyed by index into
        the exact list passed in; assignments are never withheld on low
        confidence -- the flag is the trust signal.

    Raises:
        ValueError: If fewer than 2 usable player crops exist.
        FileNotFoundError / RuntimeError: If the image is missing/undecodable.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    candidates = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.class_name == "player"
        and detection.confidence >= MIN_DETECTION_CONFIDENCE
    ]
    player_count = sum(1 for d in detections if d.class_name == "player")

    usable: list[tuple[int, np.ndarray]] = []
    for index, detection in candidates:
        try:
            crop = extract_torso_crop(image, detection)
        except ValueError:
            continue
        sig = compute_signature(crop)
        if sig.any():
            usable.append((index, sig))

    excluded_count = player_count - len(usable)
    if len(usable) < 2:
        raise ValueError(
            f"Only {len(usable)} usable player crop(s) in {image_path}; "
            f"need at least 2 to split into teams."
        )

    signatures = np.stack([sig for _, sig in usable])
    team_ids, confidences, separation = cluster_signatures(signatures)

    assignments = {
        original_index: {"team_id": team_ids[pos], "confidence": confidences[pos]}
        for pos, (original_index, _sig) in enumerate(usable)
    }
    return {
        "assignments": assignments,
        "separation_score": separation,
        "low_confidence_frame": separation < MIN_SILHOUETTE_SCORE,
        "excluded_count": excluded_count,
    }


# --------------------------------------------------------------------------
# Batch runner with cached detections + visual debug output
# --------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

#: Where cached Roboflow responses and debug renders go. Kept out of the
#: repo's own debug_output so v1 and v2 artifacts don't mix.
V2_OUTPUT_DIR = "v2_output"

TEAM_DRAW_COLORS = {0: (255, 0, 255), 1: (255, 255, 0)}  # magenta / cyan (BGR)


def cached_infer(image_path: str, cache_dir: str) -> list[Detection]:
    """infer() with a JSON disk cache so reruns don't re-hit the API."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json"
    )
    if os.path.isfile(cache_path):
        with open(cache_path) as fh:
            raw = json.load(fh)
        return [
            Detection(
                class_name=item["class_name"],
                confidence=item["confidence"],
                bbox_center=tuple(item["bbox_center"]),
                bbox_size=tuple(item["bbox_size"]),
                foot_point=tuple(item["foot_point"]),
            )
            for item in raw
        ]

    detections = infer(image_path)
    with open(cache_path, "w") as fh:
        json.dump(
            [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox_center": list(d.bbox_center),
                    "bbox_size": list(d.bbox_size),
                    "foot_point": list(d.foot_point),
                }
                for d in detections
            ],
            fh,
        )
    return detections


def render_debug(
    image: np.ndarray,
    detections: list[Detection],
    result: dict,
    annotated_path: str,
    montage_path: str,
) -> None:
    """Write an annotated frame and a per-team crop montage for eyeballing."""
    annotated = image.copy()
    for index, entry in result["assignments"].items():
        detection = detections[index]
        cx, cy = detection.bbox_center
        bw, bh = detection.bbox_size
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        color = TEAM_DRAW_COLORS[entry["team_id"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{index}:{entry['confidence']:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(annotated_path, annotated)

    tile_height = 96
    rows = []
    for team_id in (0, 1):
        tiles = []
        for index, entry in sorted(result["assignments"].items()):
            if entry["team_id"] != team_id:
                continue
            crop = extract_torso_crop(image, detections[index])
            scale = tile_height / crop.shape[0]
            tile = cv2.resize(
                crop, (max(8, int(crop.shape[1] * scale)), tile_height)
            )
            tile = cv2.copyMakeBorder(
                tile, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=TEAM_DRAW_COLORS[team_id]
            )
            cv2.putText(
                tile,
                str(index),
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        if tiles:
            max_h = max(t.shape[0] for t in tiles)
            tiles = [
                cv2.copyMakeBorder(
                    t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
                )
                for t in tiles
            ]
            rows.append(np.hstack(tiles))
    if rows:
        max_w = max(r.shape[1] for r in rows)
        rows = [
            cv2.copyMakeBorder(
                r, 0, 8, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)
            )
            for r in rows
        ]
        cv2.imwrite(montage_path, np.vstack(rows))


def run_batch(samples_dir: str, output_dir: str) -> None:
    """Run cached detection + v2 team assignment over every sample frame."""
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "detection_cache")

    image_paths = sorted(
        os.path.join(samples_dir, name)
        for name in os.listdir(samples_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_paths:
        print(f"No images found in {samples_dir}")
        return

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n--- {stem} ---")
        try:
            detections = cached_infer(image_path, cache_dir)
            result = assign_teams_v2(image_path, detections)
        except Exception as exc:  # noqa: BLE001 - reported per frame, batch continues
            print(f"FAILED: {exc}")
            continue

        team_sizes = {0: 0, 1: 0}
        for entry in result["assignments"].values():
            team_sizes[entry["team_id"]] += 1
        flag = "  [LOW CONFIDENCE]" if result["low_confidence_frame"] else ""
        print(
            f"teams {team_sizes[0]} vs {team_sizes[1]}, "
            f"excluded {result['excluded_count']}, "
            f"silhouette {result['separation_score']:.3f}{flag}"
        )
        confidences = sorted(
            (entry["confidence"], index)
            for index, entry in result["assignments"].items()
        )
        least = ", ".join(f"#{index}={conf:.2f}" for conf, index in confidences[:3])
        print(f"least confident: {least}")

        image = cv2.imread(image_path)
        render_debug(
            image,
            detections,
            result,
            os.path.join(output_dir, f"{stem}_annotated.jpg"),
            os.path.join(output_dir, f"{stem}_montage.jpg"),
        )

        with open(os.path.join(output_dir, f"{stem}_assignments.json"), "w") as fh:
            json.dump(
                {
                    "assignments": {
                        str(k): v for k, v in sorted(result["assignments"].items())
                    },
                    "separation_score": result["separation_score"],
                    "low_confidence_frame": result["low_confidence_frame"],
                    "excluded_count": result["excluded_count"],
                },
                fh,
                indent=2,
            )


if __name__ == "__main__":
    samples = sys.argv[1] if len(sys.argv) > 1 else "./Samples"
    output = sys.argv[2] if len(sys.argv) > 2 else V2_OUTPUT_DIR
    run_batch(samples, output)
