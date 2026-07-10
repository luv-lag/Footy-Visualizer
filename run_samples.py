"""Batch-runs detection + team assignment + debug_combined over a samples folder.

Standalone orchestration script: it does not add pipeline behavior of its
own, it just loops detection.infer(), team_separation.assign_teams(), and
debug_tools.debug_combined() across a directory of sample broadcast frames,
so the whole sample set can be sanity-checked in one run instead of one image
at a time.

Usage:
    python run_samples.py [samples_dir]

If samples_dir is omitted, defaults to "./Samples".
"""

from __future__ import annotations

import os
import sys

from debug_tools import debug_combined
from detection import infer
from team_separation import assign_teams

#: Recognized image extensions when scanning the samples directory.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def run_batch(samples_dir: str) -> None:
    """Run infer() + assign_teams() + debug_combined() on every image in samples_dir.

    Continues past per-image failures (e.g. one API call timing out)
    instead of aborting the whole batch, since the point of a batch debug
    run is to see results for as many frames as possible in one pass.

    Team assignment failures (e.g. fewer than 2 players in a frame) are
    treated as a soft fallback rather than a frame failure: that frame's
    combined image is still produced, just class-colored instead of
    team-colored, since detection succeeded even though team separation
    couldn't run.

    Args:
        samples_dir: Directory containing broadcast frame images.

    Raises:
        FileNotFoundError: If samples_dir does not exist.
    """
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    image_paths = sorted(
        os.path.join(samples_dir, name)
        for name in os.listdir(samples_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_paths:
        print(f"No images found in {samples_dir} (looked for {IMAGE_EXTENSIONS})")
        return

    succeeded = []
    failed = []

    for image_path in image_paths:
        print(f"\n--- {image_path} ---")
        try:
            detections = infer(image_path)
            print(f"infer() returned {len(detections)} detections")

            try:
                team_result = assign_teams(image_path, detections)
                if team_result["low_confidence_frame"]:
                    print(
                        f"Team split flagged low-confidence "
                        f"(separation score {team_result['separation_score']:.3f})"
                    )
            except ValueError as exc:
                print(f"Team assignment skipped (falling back to class coloring): {exc}")
                team_result = None

            debug_combined(image_path, detections, team_result)
            succeeded.append(image_path)
        except Exception as exc:  # noqa: BLE001 - reported per-frame, batch continues
            print(f"FAILED: {exc}")
            failed.append(image_path)

    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed out of {len(image_paths)}")
    if failed:
        print("Failed images:")
        for image_path in failed:
            print(f"  {image_path}")


if __name__ == "__main__":
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else "./Samples"
    run_batch(samples_dir)
