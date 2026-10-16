"""detect_clip.py -- precompute a clip's detection cache with a local detector.

Detection is by far the slowest stage on CPU, and it can need a different
environment (e.g. RF-DETR's torch venv) from the rest of the pipeline.
This fills ``<out_dir>/detections.json`` for every ``stride``-th frame;
``run_video.py --detector none --detect-stride <stride>`` then reads it.

    rfvenv/bin/python eval/gsr/detect_clip.py clip.mp4 out/ --size medium --stride 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))

from video.detectors import CachedDetector, RFDETRDetector  # noqa: E402
from video.reader import iter_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    cache = CachedDetector(RFDETRDetector(a.size, a.conf, a.resolution),
                           os.path.join(a.out_dir, "detections.json"))
    t0 = time.time()
    n = 0
    for idx, frame in iter_frames(a.video):
        if idx % a.stride:
            continue
        cache.detect(frame, idx)
        n += 1
        if n % 20 == 0:
            cache.save()
            print(f"frame {idx}: {(time.time() - t0) / n:.2f} s/frame", flush=True)
    cache.save()
    print(f"done: {n} frames, {(time.time() - t0) / max(n, 1):.2f} s/frame")


if __name__ == "__main__":
    main()
