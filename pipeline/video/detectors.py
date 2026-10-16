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
* :class:`RFDETRDetector` -- RF-DETR's COCO checkpoint; needs no API key,
  but only knows "person" (no goalkeeper / referee split).

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


class RFDETRDetector:
    """RF-DETR (Apache-2.0) pretrained on COCO, run locally (CPU or GPU).

    COCO has no football classes: "person" becomes "player" (goalkeepers
    and referees are not told apart) and "sports ball" becomes "ball".
    Install with ``pip install rfdetr``.

    Args:
        size: "nano", "small", "medium" or "base".
        confidence: detection threshold.
        resolution: optional inference resolution override (multiple of 32);
            broadcast players are small, so higher helps recall at a CPU cost.
    """

    name = "rfdetr"
    _COCO_PERSON, _COCO_BALL = 1, 37  # COCO category ids used by rfdetr

    def __init__(self, size: str = "medium", confidence: float = 0.35,
                 resolution: int | None = None) -> None:
        try:
            import rfdetr  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("RFDETRDetector needs the `rfdetr` package") from exc
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
               "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase}[size]
        self.model = cls(resolution=resolution) if resolution else cls()
        self.confidence = confidence

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        det = self.model.predict(frame[:, :, ::-1].copy(), threshold=self.confidence)  # BGR->RGB
        out = []
        for (x0, y0, x1, y1), conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            if cid == self._COCO_PERSON:
                name = "player"
            elif cid == self._COCO_BALL:
                name = "ball"
            else:
                continue
            w, h = float(x1 - x0), float(y1 - y0)
            cx, cy = float(x0 + w / 2), float(y0 + h / 2)
            out.append(Detection(name, float(conf), (cx, cy), (w, h), (cx, cy + h / 2)))
        return out


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
    """Build a backend by CLI name ("hosted", "local", "rfdetr", or "none")."""
    if name == "hosted":
        return RoboflowHostedDetector()
    if name == "local":
        return RoboflowLocalDetector()
    if name == "rfdetr":
        return RFDETRDetector()
    if name == "none":
        return None
    raise ValueError(f"unknown detector {name!r} (expected hosted, local or none)")
