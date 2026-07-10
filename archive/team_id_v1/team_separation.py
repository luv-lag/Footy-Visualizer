"""Team-separation stage of the football broadcast analysis pipeline.

detection.py finds every player, referee, goalkeeper, and ball in a frame and
returns pixel coordinates. It has no notion of which team a player belongs
to. This module looks at each player's jersey and splits players into two
groups (team 0 / team 1) using classical computer-vision features -- colour
distribution and local texture -- rather than a neural network.

Why classical features instead of a CNN (e.g. CLIP/ResNet embeddings): far
sideline players in a broadcast frame are often only ~20-30px wide and
motion-blurred. Colour distribution survives that degradation; a pretrained
CNN buys little on crops that small and adds real dependency/compute
overhead for no accuracy gain here.

Key realism requirement: broadcast frames vary a lot (lighting, blur,
occlusion, similar-coloured kits). This module never forces a confident split
on an unseparable frame -- it reports a separation-quality score
(`separation_score`, the silhouette score of the 2-cluster solution) and
flags frames below `MIN_SILHOUETTE_SCORE` as `low_confidence_frame`. The
assignments are still returned even when flagged (nothing is withheld) --
the flag is the honest signal for whether to trust them or fall back to
something else (e.g. spatial/pitch-position heuristics) downstream.

This module stays entirely in pixel space: no homography, no pitch
coordinates.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from detection import Detection, filter_players, infer

#: Detection-confidence pre-filter: players below this are excluded from
#: clustering entirely (a low-confidence bbox likely has a noisy/partial
#: torso crop that would just add noise to the team split).
MIN_TEAM_ASSIGNMENT_CONFIDENCE = 0.6

#: Below this silhouette score, the 2-cluster split isn't trusted -- the
#: frame is flagged low_confidence_frame rather than reporting a confident
#: wrong answer.
MIN_SILHOUETTE_SCORE = 0.25

#: Relative weight of the colour feature vs. the texture feature when
#: concatenated in extract_features. Colour is the primary signal (jerseys
#: are usually separable by colour alone); texture is a secondary signal for
#: same-colour-family kits that differ by pattern (stripes vs. plain).
COLOR_WEIGHT = 0.7
TEXTURE_WEIGHT = 0.3

#: Fraction of the player bbox height skipped from the top (head/neck) and
#: bottom (shorts/socks/boots) when isolating the torso, where jersey colour
#: is most reliable and least occluded by the ball, grass, or other players.
TORSO_TOP_FRACTION = 0.20
TORSO_BOTTOM_FRACTION = 0.40

#: Number of hue bins in the saturation-weighted colour histogram.
HUE_BINS = 32

#: Local Binary Pattern parameters (skimage 'uniform' method): P sample
#: points on a circle of radius R.
LBP_P = 8
LBP_R = 1

#: Fraction to expand each detection bbox by (each dimension) before marking
#: it "occupied" in the grass-sampling mask, so shadow/motion-blur pixels
#: right at a player's edge aren't mistakenly sampled as grass.
GRASS_MASK_DILATION = 0.12

#: Local-variance window size (pixels) for the texture score used to find
#: "confident background" (flat turf) among unoccupied pixels.
GRASS_TEXTURE_WINDOW = 9

#: Percentile (of local-variance texture score, among unoccupied pixels)
#: below which a pixel is considered "confident background." Flat turf has
#: lower local texture than crowd, advertising boards, or pitch-line paint,
#: even though those aren't inside any detection box -- this percentile cut
#: is what screens them out.
GRASS_TEXTURE_PERCENTILE = 25.0

#: Minimum number of confident-background pixels required to trust the
#: resulting grass model. Below this, estimate_grass_color_model returns
#: None (e.g. an extreme close-up frame with little or no pitch visible).
MIN_GRASS_SAMPLE_PIXELS = 500

#: Hue bins used ONLY when estimating the grass model's dominant hue peak --
#: finer than HUE_BINS since this runs once per frame (not once per crop),
#: so the extra resolution for locating the peak is free.
GRASS_HUE_ESTIMATION_BINS = 36

#: Tolerance, in real-world hue degrees (0-360 scale), around the grass
#: model's dominant hue within which a crop pixel counts as "grass-like."
#: Chosen wide enough to cover natural turf/lighting variation across a
#: pitch (shadowed vs. sunlit patches, mowing-stripe contrast) under mixed
#: stadium lighting, but narrow enough to avoid overlapping yellow kit hues
#: (~50-70 deg) on the low end or blue kit hues (~180-250 deg) on the high
#: end, given grass typically centers around 80-140 deg in real-world hue.
GRASS_HUE_TOLERANCE_DEGREES = 18.0

#: Multiplier applied to a pixel's existing (saturation-based) weight when
#: it also matches the grass model. Near-zero rather than exactly zero: a
#: smooth strong suppression rather than a hard binary cutoff, so borderline
#: pixels aren't discarded quite as abruptly as an exact-zero would.
GRASS_SUPPRESSION_FACTOR = 0.05


@dataclass
class GrassColorModel:
    """Per-frame model of what pitch grass looks like in HSV space.

    Computed ONCE PER FRAME (not once per crop) via
    estimate_grass_color_model -- turf colour and stadium lighting are
    consistent within a single frame but vary frame to frame (day game vs.
    floodlit, turf type, camera white balance), so a fixed grass-colour
    constant wouldn't generalize across frames the way a per-frame model
    does.

    Attributes:
        dominant_hue: Peak hue of confident-background pixels, in
            real-world degrees (0-360 scale).
        hue_tolerance: +/- tolerance around dominant_hue, same scale, within
            which a pixel counts as grass-like.
        sat_range: (low, high) saturation bounds, each in [0, 1], that a
            pixel's saturation must fall within to count as grass-like.
        val_range: (low, high) value/brightness bounds, each in [0, 1], that
            a pixel's value must fall within to count as grass-like.
    """

    dominant_hue: float
    hue_tolerance: float
    sat_range: tuple[float, float]
    val_range: tuple[float, float]


def estimate_grass_color_model(
    image: np.ndarray, detections: list[Detection]
) -> GrassColorModel | None:
    """Estimate what pitch grass looks like in this specific broadcast frame.

    Computed ONCE PER FRAME, not once per crop -- see GrassColorModel's
    docstring for why a per-frame model is needed instead of a fixed
    constant.

    Approach: exclude every detection's (dilated) bounding box from
    consideration -- ALL classes, not just players, since none of them are
    grass -- then among the remaining pixels take the lowest-texture
    quartile as "confident background." That texture cut specifically
    screens out crowd, advertising boards, and pitch-line paint, which all
    have more local texture than flat turf even though they aren't inside
    any detection box. The dominant hue of those confident-background
    pixels, plus a generous tolerance, becomes the grass model; saturation
    and value ranges come from the same pixels (mean +/- one std) so
    unrelated very-dark or very-saturated regions aren't wrongly folded in.

    Args:
        image: Full BGR broadcast frame.
        detections: All detections for this frame, any class.

    Returns:
        A GrassColorModel, or None if no reliable grass sample could be
        found (e.g. an extreme close-up frame with little or no pitch
        visible, or detections covering nearly the whole frame). Callers
        MUST handle None by skipping grass suppression entirely for that
        frame (pass grass_model=None to compute_color_feature) rather than
        guessing or defaulting a model.
    """
    height, width = image.shape[:2]
    occupied_mask = np.zeros((height, width), dtype=bool)

    for detection in detections:
        cx, cy = detection.bbox_center
        bbox_width, bbox_height = detection.bbox_size
        half_w = (bbox_width * (1 + GRASS_MASK_DILATION)) / 2.0
        half_h = (bbox_height * (1 + GRASS_MASK_DILATION)) / 2.0
        x1 = max(0, int(round(cx - half_w)))
        x2 = min(width, int(round(cx + half_w)))
        y1 = max(0, int(round(cy - half_h)))
        y2 = min(height, int(round(cy + half_h)))
        if x2 > x1 and y2 > y1:
            occupied_mask[y1:y2, x1:x2] = True

    unoccupied_mask = ~occupied_mask
    if not np.any(unoccupied_mask):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    window = (GRASS_TEXTURE_WINDOW, GRASS_TEXTURE_WINDOW)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=window)
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=window)
    local_variance = np.clip(mean_sq - mean * mean, 0.0, None)

    texture_threshold = float(np.percentile(local_variance[unoccupied_mask], GRASS_TEXTURE_PERCENTILE))
    confident_bg_mask = unoccupied_mask & (local_variance <= texture_threshold)

    if int(np.sum(confident_bg_mask)) < MIN_GRASS_SAMPLE_PIXELS:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bg_hue = hsv[:, :, 0][confident_bg_mask].astype(np.float64)
    bg_sat = hsv[:, :, 1][confident_bg_mask].astype(np.float64) / 255.0
    bg_val = hsv[:, :, 2][confident_bg_mask].astype(np.float64) / 255.0

    hue_histogram, _ = np.histogram(bg_hue, bins=GRASS_HUE_ESTIMATION_BINS, range=(0, 180))
    peak_bin = int(np.argmax(hue_histogram))
    dominant_hue_opencv = (peak_bin + 0.5) * (180.0 / GRASS_HUE_ESTIMATION_BINS)
    dominant_hue_degrees = dominant_hue_opencv * 2.0

    sat_mean, sat_std = float(np.mean(bg_sat)), float(np.std(bg_sat))
    val_mean, val_std = float(np.mean(bg_val)), float(np.std(bg_val))
    sat_range = (max(0.0, sat_mean - sat_std), min(1.0, sat_mean + sat_std))
    val_range = (max(0.0, val_mean - val_std), min(1.0, val_mean + val_std))

    return GrassColorModel(
        dominant_hue=dominant_hue_degrees,
        hue_tolerance=GRASS_HUE_TOLERANCE_DEGREES,
        sat_range=sat_range,
        val_range=val_range,
    )


def _load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a BGR array.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a BGR numpy array.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If the file exists but could not be decoded as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not decode image (unsupported or corrupt file): {image_path}")
    return image


def extract_torso_crop(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Crop the torso region of a player detection.

    Skips the top ~20% of the bbox (head/neck) and bottom ~40% (shorts,
    socks, boots), leaving the jersey torso -- the most reliable, least
    occluded region for team colour.

    Args:
        image: Full BGR frame the detection came from.
        detection: The player detection to crop.

    Returns:
        The cropped torso region as a BGR array (a view into `image`).

    Raises:
        ValueError: If the resulting crop would be empty, e.g. because the
            bbox is degenerate or lies entirely outside the image bounds.
    """
    height, width = image.shape[:2]
    cx, cy = detection.bbox_center
    bw, bh = detection.bbox_size

    x1 = max(0, min(int(round(cx - bw / 2)), width))
    x2 = max(0, min(int(round(cx + bw / 2)), width))
    y1 = max(0, min(int(round(cy - bh / 2)), height))
    y2 = max(0, min(int(round(cy + bh / 2)), height))

    box_height = y2 - y1
    torso_y1 = max(0, min(y1 + int(round(TORSO_TOP_FRACTION * box_height)), height))
    torso_y2 = max(0, min(y2 - int(round(TORSO_BOTTOM_FRACTION * box_height)), height))

    if torso_y2 <= torso_y1 or x2 <= x1:
        raise ValueError(
            f"Degenerate torso crop for detection with bbox_center={detection.bbox_center}, "
            f"bbox_size={detection.bbox_size}: crop would be empty after clamping to image "
            f"bounds ({width}x{height})"
        )

    return image[torso_y1:torso_y2, x1:x2]


def compute_color_feature(
    crop: np.ndarray,
    return_weights: bool = False,
    grass_model: GrassColorModel | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute a saturation-weighted hue histogram for a torso crop.

    This is a *distribution* over hue, not a single mean hue, so striped or
    two-tone kits (which would average out to a meaningless mid-hue) produce
    a distinct, separable signature instead. Pixels are weighted by their
    saturation so near-grey/white/black pixels (shadow, motion blur, grass
    bleed-through) contribute little to the signal, while saturated jersey
    colour dominates.

    Args:
        crop: BGR torso crop, as returned by extract_torso_crop.
        return_weights: If True, also return the per-pixel EFFECTIVE weight
            map used to build the histogram -- i.e. after grass suppression
            has already been applied, if a grass_model was given (same
            height/width as crop, values in [0, 1]). Added for diagnostic
            tooling (team_debug.py) that needs to inspect or visualize
            exactly which pixels drove the colour signal; existing callers
            are unaffected since this defaults to False.
        grass_model: Optional per-frame GrassColorModel (see
            estimate_grass_color_model). When given, pixels whose hue,
            saturation, AND value all fall inside the model's grass ranges
            get an ADDITIONAL near-zero weight multiplier -- a second,
            independent suppression layer alongside the existing
            saturation-based weighting; neither overrides the other, both
            reduce a pixel's contribution. This targets grass bleed-through
            that saturation weighting alone can miss: stadium-lit grass can
            be moderately saturated (not reliably downweighted by
            saturation alone), while a dark/desaturated kit already is --
            without this, grass can end up weighted HIGHER than the
            player's own jersey. None (default) preserves the exact prior
            behaviour.

    Returns:
        The L2-normalized histogram of length HUE_BINS (all-zero if the crop
        has no effective colour signal). If return_weights is True, returns
        (histogram, weight_map) instead.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float64)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    weight_map = saturation.copy()

    if grass_model is not None:
        value = hsv[:, :, 2].astype(np.float64) / 255.0

        model_hue_opencv = grass_model.dominant_hue / 2.0
        model_tolerance_opencv = grass_model.hue_tolerance / 2.0
        hue_diff = np.abs(hue - model_hue_opencv)
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)  # hue is circular

        is_grass_like = (
            (hue_diff <= model_tolerance_opencv)
            & (saturation >= grass_model.sat_range[0])
            & (saturation <= grass_model.sat_range[1])
            & (value >= grass_model.val_range[0])
            & (value <= grass_model.val_range[1])
        )
        weight_map = np.where(is_grass_like, weight_map * GRASS_SUPPRESSION_FACTOR, weight_map)

    histogram, _ = np.histogram(hue.ravel(), bins=HUE_BINS, range=(0, 180), weights=weight_map.ravel())

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm

    if return_weights:
        return histogram, weight_map
    return histogram


def compute_texture_feature(crop: np.ndarray) -> np.ndarray:
    """Compute a Local Binary Pattern histogram for a torso crop.

    Captures pattern differences (stripes, hoops, plain) that colour alone
    can't distinguish -- useful when two teams share a similar base colour.

    Note: this signal naturally weakens on very small or motion-blurred
    crops, since LBP depends on local pixel-level structure that blur
    destroys. That's expected and acceptable -- the colour feature carries
    those cases; texture is a secondary, not primary, signal.

    Args:
        crop: BGR torso crop, as returned by extract_torso_crop.

    Returns:
        An L2-normalized histogram of length LBP_P + 2 (the standard
        'uniform' LBP histogram size).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")

    n_bins = LBP_P + 2
    histogram, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    histogram = histogram.astype(np.float64)

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm
    return histogram


def extract_features(
    crops: list[np.ndarray], grass_model: GrassColorModel | None = None
) -> np.ndarray:
    """Build the combined colour + texture feature vector for each crop.

    Kept generic (crops in, feature matrix out) so the underlying feature
    method could be swapped later without changing callers.

    Args:
        crops: Torso crops, as returned by extract_torso_crop.
        grass_model: Optional per-frame GrassColorModel, forwarded to every
            compute_color_feature call. Compute this ONCE per frame (via
            estimate_grass_color_model) and pass the same instance in for
            every crop from that frame -- do not recompute per crop.

    Returns:
        Array of shape (n_crops, feature_dim), where feature_dim =
        HUE_BINS + (LBP_P + 2).

    Raises:
        ValueError: If `crops` is empty, or any individual crop is empty.
    """
    if not crops:
        raise ValueError("extract_features called with an empty list of crops")

    rows = []
    for index, crop in enumerate(crops):
        if crop.size == 0:
            raise ValueError(f"Crop at index {index} is empty, cannot extract features")
        color_feature = compute_color_feature(crop, grass_model=grass_model) * COLOR_WEIGHT
        texture_feature = compute_texture_feature(crop) * TEXTURE_WEIGHT
        rows.append(np.concatenate([color_feature, texture_feature]))

    return np.stack(rows, axis=0)


def cluster_teams(features: np.ndarray) -> tuple[list[int], list[float], float]:
    """Split feature vectors into two teams via KMeans.

    Args:
        features: Array of shape (n_players, feature_dim), as returned by
            extract_features.

    Returns:
        A tuple (team_ids, confidences, separation_score):
        - team_ids: 0/1 cluster label per input row, in input order.
        - confidences: per-player confidence in [0, 1], computed as
          1 / (1 + distance_to_assigned_centroid). A distance of 0 (the
          player's feature vector exactly matches its team's centroid)
          gives confidence 1.0; confidence falls off smoothly as the
          player's features drift from its assigned centroid. This is a
          per-player measure of how clearly *that player* fits its cluster,
          independent of the frame-level separation_score.
        - separation_score: silhouette score of the 2-cluster solution,
          in [-1, 1] (higher = cleaner separation between the two teams).
          Returns -1.0 in the degenerate case where KMeans collapses both
          clusters into a single label (e.g. all feature vectors identical),
          since silhouette score is undefined there and -1.0 reliably
          triggers the low_confidence_frame flag downstream.

    Raises:
        ValueError: If fewer than 2 feature rows are given.
    """
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to cluster into 2 teams, got {n_samples}")

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    distances = np.linalg.norm(features - centers[labels], axis=1)
    confidences = (1.0 / (1.0 + distances)).tolist()

    if len(set(labels.tolist())) < 2:
        separation_score = -1.0
    else:
        separation_score = float(silhouette_score(features, labels))

    return labels.tolist(), confidences, separation_score


def assign_teams(image_path: str, detections: list[Detection]) -> dict:
    """Split detected players into two teams for a single broadcast frame.

    Orchestrates the full stage: filter to players, apply the confidence
    pre-filter, extract torso crops, estimate this frame's grass colour
    model (once, from all detections), extract features with grass
    suppression applied, cluster.

    Args:
        image_path: Path to the broadcast frame the detections came from.
        detections: Full detection list, as returned by infer() -- i.e. the
            ORIGINAL list including non-player classes, not a pre-filtered
            one.

    Returns:
        A dict:
        {
            "assignments": {
                <index into the ORIGINAL detections list>: {
                    "team_id": 0 or 1,
                    "confidence": float in [0, 1],
                },
                ...
            },
            "separation_score": float,       # silhouette score, see cluster_teams
            "low_confidence_frame": bool,    # True if separation_score < MIN_SILHOUETTE_SCORE
            "excluded_count": int,           # players dropped by the confidence pre-filter
        }
        Assignments are keyed by index into the exact `detections` list
        passed in, so callers (e.g. debug_tools.py) can look them up
        directly against the same list used elsewhere in the pipeline.
        Non-player detections and players dropped by the confidence
        pre-filter have no entry.

        IMPORTANT: when low_confidence_frame is True, assignments are still
        returned in full (never withheld) -- the flag is the signal that
        they should not be trusted, not a reason to hide them.

    Raises:
        ValueError: If fewer than 2 players survive the confidence
            pre-filter (can't cluster into 2 groups), or if crop/feature
            extraction produces degenerate data.
        FileNotFoundError: If image_path does not exist.
        RuntimeError: If image_path exists but can't be decoded.
    """
    players = filter_players(detections)
    player_object_ids = {id(player) for player in players}
    indexed_players = [
        (index, detection)
        for index, detection in enumerate(detections)
        if id(detection) in player_object_ids
    ]

    filtered_players = [
        (index, detection)
        for index, detection in indexed_players
        if detection.confidence >= MIN_TEAM_ASSIGNMENT_CONFIDENCE
    ]
    excluded_count = len(indexed_players) - len(filtered_players)

    if len(filtered_players) < 2:
        raise ValueError(
            f"Only {len(filtered_players)} player(s) survived the confidence pre-filter "
            f"(>= {MIN_TEAM_ASSIGNMENT_CONFIDENCE}); need at least 2 to cluster into 2 teams."
        )

    image = _load_image(image_path)

    # Computed ONCE for the whole frame, from ALL detections (any class) --
    # not per crop -- so every player's colour feature suppresses grass
    # consistently. estimate_grass_color_model returns None if no reliable
    # sample was found (e.g. an extreme close-up); extract_features/
    # compute_color_feature already treat None as "skip grass suppression."
    grass_model = estimate_grass_color_model(image, detections)

    crops = []
    for original_index, detection in filtered_players:
        try:
            crops.append(extract_torso_crop(image, detection))
        except ValueError as exc:
            raise ValueError(
                f"Failed to extract torso crop for detection at original index "
                f"{original_index}: {exc}"
            ) from exc

    features = extract_features(crops, grass_model=grass_model)
    team_ids, confidences, separation_score = cluster_teams(features)

    assignments = {
        original_index: {"team_id": team_ids[position], "confidence": confidences[position]}
        for position, (original_index, _detection) in enumerate(filtered_players)
    }

    return {
        "assignments": assignments,
        "separation_score": separation_score,
        "low_confidence_frame": separation_score < MIN_SILHOUETTE_SCORE,
        "excluded_count": excluded_count,
    }


if __name__ == "__main__":
    # Replace this placeholder with a real broadcast frame path, or pass one
    # as a command-line argument: `python team_separation.py path/to/frame.jpg`
    DEFAULT_IMAGE_PATH = "sample_frame.jpg"

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    all_detections = infer(image_path)
    result = assign_teams(image_path, all_detections)

    team_counts = Counter(entry["team_id"] for entry in result["assignments"].values())

    print(f"Players assigned: {len(result['assignments'])}")
    for team_id in sorted(team_counts):
        print(f"  team {team_id}: {team_counts[team_id]} players")
    print(f"Excluded by confidence pre-filter: {result['excluded_count']}")
    print(f"Separation score (silhouette): {result['separation_score']:.3f}")

    if result["low_confidence_frame"]:
        print("!" * 60)
        print("LOW CONFIDENCE FRAME -- team split is NOT reliable for this frame")
        print(f"(separation score below threshold {MIN_SILHOUETTE_SCORE})")
        print("!" * 60)
    else:
        print("Frame passed the separation-quality threshold.")
