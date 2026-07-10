"""Diagnostic tool for the team-separation colour/texture features.

team_separation.py assigns players to teams via a saturation-weighted hue
histogram (colour) plus an LBP histogram (texture), then KMeans. This module
does not change any of that -- it's a read-only diagnostic that lets you
CONFIRM OR RULE OUT specific failure hypotheses with actual numbers and
visuals, for two suspected problems:

  1. Green-kit contamination: a team wearing green (e.g. a Betis-style kit)
     may have pitch grass bleeding into the torso crop, skewing their hue
     histogram toward green regardless of their actual kit colour.
  2. Dark/navy-kit downweighting: darker blues are often lower-saturation
     than bright kit colours, so the saturation-weighting scheme may be
     downweighting the kit itself and letting non-kit pixels (skin, logos,
     background bleed) dominate the histogram instead.

debug_team_features(image_path, detections) produces:
  - A visual crop grid (JPEG): one row per player, sorted by ascending
    team-assignment confidence (least confident / most suspicious first),
    showing the raw torso crop, a heatmap of the per-pixel saturation
    weight that actually went into the colour histogram, and a swatch of
    the histogram's dominant colour.
  - A numeric table (printed + CSV): per-player hue/saturation/value/
    low-saturation-fraction stats, plus a per-team aggregate mean effective
    (weighted) saturation -- the blue-kit hypothesis in one number.

Note on the "low-saturation" fraction: compute_color_feature does not use a
hard cutoff internally (it weights every pixel continuously by saturation).
LOW_SATURATION_THRESHOLD below is a fixed threshold defined ONLY for this
diagnostic, to give a readable "fraction of near-background pixels" number.
It's a proxy for "contributed little," not a literal internal constant.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from debug_tools import TEAM_COLORS, TEXT_COLOR, _ensure_output_dir, _load_image, _make_section_header, _scale_factor
from detection import Detection, infer
from team_separation import (
    HUE_BINS,
    assign_teams,
    compute_color_feature,
    estimate_grass_color_model,
    extract_torso_crop,
)

#: Diagnostic-only cutoff (0-1 saturation scale) below which a pixel is
#: counted as "low-saturation/background" for the reported fraction. Not
#: used internally by compute_color_feature, which weights continuously --
#: see module docstring.
LOW_SATURATION_THRESHOLD = 0.15


def _compute_pixel_stats(crop: np.ndarray, weight_map: np.ndarray, histogram: np.ndarray) -> dict:
    """Compute per-crop colour statistics used by both outputs.

    Args:
        crop: BGR torso crop.
        weight_map: Per-pixel saturation weight (crop's H x W, in [0, 1]),
            as returned by compute_color_feature(crop, return_weights=True).
        histogram: The corresponding normalized hue histogram.

    Returns:
        A dict with dominant_hue_degrees (None if no colour signal),
        mean_saturation_unweighted, mean_saturation_weighted, mean_value,
        fraction_low_saturation, has_color_signal, and peak_bin (None if no
        colour signal).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    value = hsv[:, :, 2].astype(np.float64) / 255.0

    mean_saturation_unweighted = float(np.mean(saturation))
    mean_value = float(np.mean(value))
    fraction_low_saturation = float(np.mean(weight_map < LOW_SATURATION_THRESHOLD))

    total_weight = float(np.sum(weight_map))
    has_color_signal = total_weight > 1e-9

    if has_color_signal:
        # Self-weighted mean saturation: pixels that contributed more to the
        # histogram (higher saturation) count more here too, so this is
        # literally "the saturation level the histogram was built from,"
        # not just the crop's raw average.
        mean_saturation_weighted = float(np.sum(weight_map * weight_map) / total_weight)
        peak_bin = int(np.argmax(histogram))
        dominant_hue_degrees = (peak_bin + 0.5) * (180.0 / HUE_BINS) * 2.0
    else:
        mean_saturation_weighted = 0.0
        peak_bin = None
        dominant_hue_degrees = None

    return {
        "has_color_signal": has_color_signal,
        "peak_bin": peak_bin,
        "dominant_hue_degrees": dominant_hue_degrees,
        "mean_saturation_unweighted": mean_saturation_unweighted,
        "mean_saturation_weighted": mean_saturation_weighted,
        "mean_value": mean_value,
        "fraction_low_saturation": fraction_low_saturation,
    }


def _dominant_color_swatch(record: dict) -> tuple[int, int, int]:
    """Build a representative BGR swatch color for a player's peak hue bin.

    Uses the crop's actual effective saturation/value (not full-saturation),
    so a genuinely muted/dark kit shows as a muted/dark swatch -- exactly
    the signal needed to see whether a dark-blue kit's histogram peak is
    still visually "kit-like" or has been swamped by something else.

    Args:
        record: A per-player record as built in debug_team_features,
            containing has_color_signal, peak_bin, mean_saturation_weighted,
            mean_value.

    Returns:
        A BGR color tuple. Neutral grey if the crop had no colour signal.
    """
    if not record["has_color_signal"]:
        return (128, 128, 128)

    hue_opencv = (record["peak_bin"] + 0.5) * (180.0 / HUE_BINS)
    saturation = float(np.clip(record["mean_saturation_weighted"], 0.0, 1.0)) * 255
    value = float(np.clip(record["mean_value"], 0.0, 1.0)) * 255

    hsv_pixel = np.uint8([[[hue_opencv, saturation, value]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _weight_heatmap_overlay(crop: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    """Overlay the per-pixel saturation weight as a heatmap on the crop.

    Hot (red) regions are pixels that strongly influenced the colour
    histogram; cool (blue) regions were mostly ignored. This is the direct
    visual test for both hypotheses: grass bleed-through showing up "hot"
    confirms green contamination, and a kit region showing up "cool"
    confirms the kit itself is being downweighted.

    Args:
        crop: BGR torso crop.
        weight_map: Per-pixel saturation weight, in [0, 1].

    Returns:
        A BGR image the same size as crop, blending the original pixels
        with a JET colormap of the weight.
    """
    weight_uint8 = np.clip(weight_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(weight_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop, 0.5, heatmap, 0.5, 0)


def _format_row_text(record: dict) -> str:
    """Format the per-player stats line shown next to each crop-grid row."""
    hue_text = f"{record['dominant_hue_degrees']:.0f}deg" if record["has_color_signal"] else "n/a"
    return (
        f"idx {record['index']}  team {record['team_id']}  conf {record['confidence']:.2f}  "
        f"hue {hue_text}  sat_w {record['mean_saturation_weighted']:.2f}  "
        f"sat_raw {record['mean_saturation_unweighted']:.2f}  "
        f"val {record['mean_value']:.2f}  low_sat {record['fraction_low_saturation']:.0%}"
    )


def _build_feature_row(
    record: dict,
    text: str,
    cell_w: int,
    cell_h: int,
    gap: int,
    text_col_width: int,
    font: int,
    font_scale: float,
    thickness: int,
    text_padding: int,
    scale: float,
) -> np.ndarray:
    """Build one bordered row: raw crop | weight heatmap | swatch | stats text.

    All size parameters are shared across every row in the grid (computed
    once by the caller) so every row has identical width and cv2.vconcat
    can stack them directly.
    """
    crop = record["crop"]
    weight_map = record["weight_map"]

    raw_resized = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    heatmap_resized = cv2.resize(
        _weight_heatmap_overlay(crop, weight_map), (cell_w, cell_h), interpolation=cv2.INTER_AREA
    )
    swatch = np.full((cell_h, cell_h, 3), _dominant_color_swatch(record), dtype=np.uint8)
    gap_col = np.zeros((cell_h, gap, 3), dtype=np.uint8)

    images_block = cv2.hconcat([raw_resized, gap_col, heatmap_resized, gap_col, swatch])

    text_block = np.zeros((cell_h, text_col_width, 3), dtype=np.uint8)
    (_text_width, text_height), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_y = cell_h // 2 + text_height // 2
    cv2.putText(text_block, text, (text_padding, text_y), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    row = cv2.hconcat([images_block, gap_col, text_block])

    border = max(2, round(3 * scale))
    team_color = TEAM_COLORS.get(record["team_id"], (200, 200, 200))
    return cv2.copyMakeBorder(row, border, border, border, border, cv2.BORDER_CONSTANT, value=team_color)


def _save_crop_grid(records: list[dict], skipped: list[dict], scale: float, timestamp: str) -> str:
    """Assemble and save the visual crop grid (Output A).

    Args:
        records: Per-player records, already sorted by ascending confidence.
        skipped: Records for players whose crop couldn't be extracted.
        scale: Resolution-relative scale factor from the source frame.
        timestamp: Shared timestamp string so this file pairs with the CSV
            from the same run.

    Returns:
        The path the grid image was saved to.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, f"team_features_{timestamp}.jpg")

    if not records:
        width = max(400, round(800 * scale))
        placeholder = _make_section_header(width, "No players available for feature diagnostic", scale)
        cv2.imwrite(output_path, placeholder)
        return output_path

    cell_h = max(60, round(110 * scale))
    cell_w = max(40, round(70 * scale))
    gap = max(4, round(8 * scale))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.42 * scale)
    thickness = max(1, round(1.2 * scale))
    text_padding = max(6, round(10 * scale))

    texts = [_format_row_text(record) for record in records]
    max_text_width = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in texts)
    text_col_width = max_text_width + 2 * text_padding

    row_images = [
        _build_feature_row(
            record, text, cell_w, cell_h, gap, text_col_width, font, font_scale, thickness, text_padding, scale
        )
        for record, text in zip(records, texts)
    ]

    grid_width = row_images[0].shape[1]
    header = _make_section_header(
        grid_width,
        "TEAM COLOR FEATURES -- sorted by ascending confidence (most suspicious first)",
        scale,
    )
    parts = [header, *row_images]

    if skipped:
        skip_text = "Skipped (degenerate crop): " + ", ".join(f"idx {s['index']}" for s in skipped)
        parts.append(_make_section_header(grid_width, skip_text, scale))

    grid = cv2.vconcat(parts)
    cv2.imwrite(output_path, grid)
    return output_path


def _save_and_print_table(records: list[dict], skipped: list[dict], timestamp: str) -> str:
    """Print the numeric table to console and save it as CSV (Output B).

    Also prints, per team, the mean effective (weighted) saturation --
    the single number that directly surfaces the dark-kit hypothesis: if
    one team's mean is much lower than the other's, that team's histogram
    is being built from systematically less colour signal.

    Args:
        records: Per-player records, already sorted by ascending confidence.
        skipped: Records for players whose crop couldn't be extracted.
        timestamp: Shared timestamp string so this file pairs with the JPEG
            from the same run.

    Returns:
        The path the CSV was saved to.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, f"team_features_{timestamp}.csv")

    fieldnames = [
        "player_index",
        "team_id",
        "confidence",
        "dominant_hue_degrees",
        "mean_saturation_unweighted",
        "mean_saturation_weighted",
        "mean_value",
        "fraction_low_saturation",
        "skipped_reason",
    ]

    print("\n" + "=" * 100)
    print(
        f"{'idx':>4} {'team':>4} {'conf':>6} {'hue_deg':>8} {'sat_raw':>8} "
        f"{'sat_wtd':>8} {'value':>7} {'low_sat_%':>10}"
    )
    print("-" * 100)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            hue_display = f"{record['dominant_hue_degrees']:.1f}" if record["has_color_signal"] else "n/a"
            print(
                f"{record['index']:>4} {record['team_id']:>4} {record['confidence']:>6.2f} "
                f"{hue_display:>8} {record['mean_saturation_unweighted']:>8.3f} "
                f"{record['mean_saturation_weighted']:>8.3f} {record['mean_value']:>7.3f} "
                f"{record['fraction_low_saturation'] * 100:>9.1f}%"
            )
            writer.writerow(
                {
                    "player_index": record["index"],
                    "team_id": record["team_id"],
                    "confidence": record["confidence"],
                    "dominant_hue_degrees": record["dominant_hue_degrees"] if record["has_color_signal"] else "",
                    "mean_saturation_unweighted": record["mean_saturation_unweighted"],
                    "mean_saturation_weighted": record["mean_saturation_weighted"],
                    "mean_value": record["mean_value"],
                    "fraction_low_saturation": record["fraction_low_saturation"],
                    "skipped_reason": "",
                }
            )

        for skip in skipped:
            print(f"SKIPPED idx {skip['index']} (team {skip['team_id']}): {skip['reason']}")
            writer.writerow(
                {
                    "player_index": skip["index"],
                    "team_id": skip["team_id"],
                    "confidence": skip["confidence"],
                    "dominant_hue_degrees": "",
                    "mean_saturation_unweighted": "",
                    "mean_saturation_weighted": "",
                    "mean_value": "",
                    "fraction_low_saturation": "",
                    "skipped_reason": skip["reason"],
                }
            )

    print("-" * 100)
    for team_id in sorted({record["team_id"] for record in records}):
        team_records = [record for record in records if record["team_id"] == team_id]
        mean_weighted_sat = sum(r["mean_saturation_weighted"] for r in team_records) / len(team_records)
        print(f"Team {team_id}: mean effective (weighted) saturation = {mean_weighted_sat:.3f}  (n={len(team_records)})")
    print("=" * 100 + "\n")

    return output_path


def debug_team_features(image_path: str, detections: list[Detection]) -> None:
    """Diagnose the colour/texture features behind a frame's team split.

    Runs assign_teams() internally, then for every player that survived its
    confidence pre-filter, extracts the torso crop and the exact per-pixel
    saturation weight map that fed compute_color_feature's histogram.
    Produces a visual crop grid and a numeric table (see module docstring).

    Args:
        image_path: Path to the broadcast frame.
        detections: Full detection list, as returned by infer().

    Side effects:
        Writes ./debug_output/team_features_<timestamp>.jpg and
        ./debug_output/team_features_<timestamp>.csv, and prints the numeric
        table plus per-team aggregate saturation to the console.

    Raises:
        ValueError: Propagated from assign_teams() if fewer than 2 players
            survive its confidence pre-filter -- there's nothing to diagnose
            without a team split to inspect.
    """
    image = _load_image(image_path)
    scale = _scale_factor(image)
    result = assign_teams(image_path, detections)

    # Mirrors assign_teams()'s own internal call: same image + same
    # detections deterministically gives the same model. Computed once here
    # (not per crop) so this diagnostic's heatmap/stats reflect the exact
    # grass suppression assign_teams() actually applied for this frame.
    grass_model = estimate_grass_color_model(image, detections)
    print(f"Grass color model: {'available -- suppression active' if grass_model is not None else 'NOT available -- no grass suppression applied (see estimate_grass_color_model)'}")

    print(
        f"assign_teams: separation_score={result['separation_score']:.3f}  "
        f"low_confidence_frame={result['low_confidence_frame']}  "
        f"excluded_count={result['excluded_count']}"
    )

    records = []
    skipped = []

    for index, entry in result["assignments"].items():
        detection = detections[index]
        try:
            crop = extract_torso_crop(image, detection)
        except ValueError as exc:
            skipped.append(
                {"index": index, "team_id": entry["team_id"], "confidence": entry["confidence"], "reason": str(exc)}
            )
            continue

        if crop.size == 0:
            skipped.append(
                {
                    "index": index,
                    "team_id": entry["team_id"],
                    "confidence": entry["confidence"],
                    "reason": "empty crop (zero-size after extraction)",
                }
            )
            continue

        histogram, weight_map = compute_color_feature(crop, return_weights=True, grass_model=grass_model)
        stats = _compute_pixel_stats(crop, weight_map, histogram)
        records.append(
            {
                "index": index,
                "team_id": entry["team_id"],
                "confidence": entry["confidence"],
                "crop": crop,
                "weight_map": weight_map,
                **stats,
            }
        )

    records.sort(key=lambda record: record["confidence"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = _save_crop_grid(records, skipped, scale, timestamp)
    csv_path = _save_and_print_table(records, skipped, timestamp)

    print(f"Saved team-feature crop grid to {grid_path}")
    print(f"Saved team-feature table (CSV) to {csv_path}")


if __name__ == "__main__":
    # Replace this placeholder with a real broadcast frame path, or pass one
    # as a command-line argument: `python team_debug.py path/to/frame.jpg`
    DEFAULT_IMAGE_PATH = "sample_frame.jpg"

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    all_detections = infer(image_path)
    result = assign_teams(image_path, all_detections)
    print(f"assign_teams separation_score: {result['separation_score']:.3f}")

    debug_team_features(image_path, all_detections)
