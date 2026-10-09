"""detectors.py -- interchangeable person/ball detectors for video frames.

Every backend is a callable ``frame (BGR ndarray) -> list[Detection]``
returning the same :class:`detection.Detection` records as the
single-frame pipeline, so everything downstream is backend-agnostic.

Backends:

* :class:`RoboflowHostedDetector` -- the model the single-frame pipeline
  already uses, over Roboflow's hosted API.  No local compute, but one
  network round-trip per call: use with ``--detect-stride`` > 1 on video.
* :class:`RoboflowLocalDetector` -- the *same* model run locally through
  Roboflow's ``inference`` package (CPU or GPU; weights are downloaded once
  with the API key).  Optional dependency, imported lazily.

:class:`CachedDetector` wraps any backend with a per-video JSON cache keyed
by frame index, so re-running the pipeline (tuning tracking / rendering)
never re-detects.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable

import numpy as np

from detection import MODEL_ID, Detection, _get_api_key, _parse_prediction, infer_frame

DetectFn = Callable[[np.ndarray], list[Detection]]


class RoboflowHostedDetector:
    """Roboflow hosted inference (network-bound; no GPU needed)."""

    name = "roboflow-hosted"

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        return infer_frame(frame)


class RoboflowLocalDetector:
    """The same Roboflow model, run locally via the ``inference`` package.

    Install with ``pip install inference`` (CPU) or ``inference-gpu``.
    """

    name = "roboflow-local"

    def __init__(self, model_id: str = MODEL_ID, confidence: float = 0.3) -> None:
        try:
            from inference import get_model  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RoboflowLocalDetector needs the `inference` package "
                "(pip install inference, or inference-gpu for CUDA)"
            ) from exc
        self.model = get_model(model_id=model_id, api_key=_get_api_key())
        self.confidence = confidence

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.infer(frame, confidence=self.confidence)[0]
        return [
            _parse_prediction(
                {
                    "class": p.class_name,
                    "confidence": p.confidence,
                    "x": p.x,
                    "y": p.y,
                    "width": p.width,
                    "height": p.height,
                }
            )
            for p in result.predictions
        ]


class CachedDetector:
    """Per-video detection cache (JSON, keyed by frame index).

    Args:
        inner: the detector to call on a cache miss, or None for a
            cache-only run (missing frames then return no detections).
        path: JSON cache file, e.g. ``<out>/detections.json``.
    """

    def __init__(self, inner: DetectFn | None, path: str) -> None:
        self.inner = inner
        self.path = path
        self._cache: dict[str, list[dict]] = {}
        if os.path.isfile(path):
            with open(path) as f:
                self._cache = json.load(f)
        self._dirty = False

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        key = str(frame_idx)
        if key not in self._cache:
            if self.inner is None:
                return []
            self._cache[key] = [asdict(d) for d in self.inner(frame)]
            self._dirty = True
        return [
            Detection(
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox_center=tuple(d["bbox_center"]),
                bbox_size=tuple(d["bbox_size"]),
                foot_point=tuple(d["foot_point"]),
            )
            for d in self._cache[key]
        ]

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.path)
        self._dirty = False


def make_detector(name: str) -> DetectFn | None:
    """Build a backend by CLI name ("hosted", "local", or "none")."""
    if name == "hosted":
        return RoboflowHostedDetector()
    if name == "local":
        return RoboflowLocalDetector()
    if name == "none":
        return None
    raise ValueError(f"unknown detector {name!r} (expected hosted, local or none)")
