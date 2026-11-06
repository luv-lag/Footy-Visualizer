"""Detection stage for the football broadcast analysis pipeline.

This module is responsible for exactly one thing: calling the Roboflow
inference API on a single broadcast frame and returning structured,
pixel-space detection data (ball, player, referee, goalkeeper).

It deliberately knows nothing about team assignment, homography /
pitch-coordinate conversion, or visualization -- those are separate stages
built independently elsewhere in the pipeline. Downstream stages depend on
the `Detection` shape defined here, so this module should stay small and
stable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

#: Roboflow model identifier: workspace-model-slug/version.
MODEL_ID = "football-players-detection-3zvbc/19"

#: Roboflow's hosted inference endpoint.
API_URL = "https://detect.roboflow.com"

#: Fields Roboflow is expected to return for each prediction.
_REQUIRED_PREDICTION_FIELDS = ("class", "confidence", "x", "y", "width", "height")


@dataclass
class Detection:
    """A single detected object in image (pixel) coordinates.

    Attributes:
        class_name: One of "ball", "player", "referee", "goalkeeper".
        confidence: Model confidence score in [0, 1].
        bbox_center: (x, y) pixel coordinates of the bounding box center,
            as returned directly by Roboflow.
        bbox_size: (width, height) of the bounding box in pixels.
        foot_point: (x, y) pixel coordinates of the point directly below
            the bbox center, at the bottom edge of the box -- i.e.
            ``(x, y + height / 2)``. This approximates where the detected
            object touches the pitch, which later pipeline stages use for
            homography-based pitch-coordinate conversion.
    """

    class_name: str
    confidence: float
    bbox_center: tuple[float, float]
    bbox_size: tuple[float, float]
    foot_point: tuple[float, float]


def _get_api_key() -> str:
    """Load ``.env`` from the current working directory and return the API key.

    Returns:
        The value of the ROBOFLOW_API_KEY environment variable.

    Raises:
        RuntimeError: If ROBOFLOW_API_KEY is not set after loading ``.env``.
    """
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Set ROBOFLOW_API_KEY in a .env file "
            "in this directory, e.g.:\n\n"
            "    ROBOFLOW_API_KEY=your_key_here\n"
        )
    return api_key


def _parse_prediction(prediction: dict) -> Detection:
    """Convert a single raw Roboflow prediction dict into a Detection.

    Args:
        prediction: One entry from the Roboflow response's "predictions" list.

    Returns:
        The parsed Detection.

    Raises:
        ValueError: If the prediction is missing any required field.
    """
    missing = [field for field in _REQUIRED_PREDICTION_FIELDS if field not in prediction]
    if missing:
        raise ValueError(
            f"Malformed Roboflow prediction, missing field(s) {missing}: {prediction!r}"
        )

    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    return Detection(
        class_name=str(prediction["class"]),
        confidence=float(prediction["confidence"]),
        bbox_center=(x, y),
        bbox_size=(width, height),
        foot_point=(x, y + height / 2),
    )


def infer(image_path: str) -> list[Detection]:
    """Run detection on a single image via the Roboflow inference API.

    Calls the "football-players-detection-3zvbc/19" model and returns every
    detected object across all four classes (ball, player, referee,
    goalkeeper), fully parsed into `Detection` objects. This function does
    not filter by class -- use `filter_players` (or a similar helper) for
    that.

    Args:
        image_path: Path to a local image file to run inference on.

    Returns:
        A list of `Detection` objects, one per object detected in the frame.
        May be empty if the model genuinely finds nothing in the frame.

    Raises:
        RuntimeError: If ROBOFLOW_API_KEY is missing/invalid, if the API
            call fails (e.g. network failure), or if the response is
            malformed in a way that prevents parsing.
        FileNotFoundError: If `image_path` does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return _infer_any(image_path)


def infer_frame(frame: np.ndarray) -> list[Detection]:
    """Like :func:`infer`, but for an in-memory BGR frame (e.g. from a video).

    Args:
        frame: BGR uint8 image array.

    Returns:
        A list of `Detection` objects, one per object detected in the frame.

    Raises:
        RuntimeError: Same conditions as :func:`infer`.
    """
    return _infer_any(frame)


def _infer_any(image: str | np.ndarray) -> list[Detection]:
    """Shared body of :func:`infer` / :func:`infer_frame`."""
    api_key = _get_api_key()
    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)

    try:
        result = client.infer(image, model_id=MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(
            f"Roboflow inference request failed for model {MODEL_ID!r}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected Roboflow response type {type(result).__name__}, expected dict: {result!r}"
        )

    if "predictions" not in result:
        raise RuntimeError(
            f"Malformed Roboflow response, missing 'predictions' key: {result!r}"
        )

    predictions = result["predictions"]
    if not isinstance(predictions, list):
        raise RuntimeError(
            f"Malformed Roboflow response, 'predictions' is not a list: {predictions!r}"
        )

    return [_parse_prediction(prediction) for prediction in predictions]


def filter_players(detections: list[Detection]) -> list[Detection]:
    """Return only the detections classified as "player".

    Kept separate from `infer` so that other class-specific filters
    (goalkeeper, referee, ball) can be added later without touching the
    inference logic.

    Args:
        detections: Detections as returned by `infer`.

    Returns:
        The subset of `detections` with `class_name == "player"`.
    """
    return [detection for detection in detections if detection.class_name == "player"]


if __name__ == "__main__":
    # Replace this placeholder with a real broadcast frame path, or pass one
    # as a command-line argument: `python detection.py path/to/frame.jpg`
    DEFAULT_IMAGE_PATH = "sample_frame.jpg"

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    all_detections = infer(image_path)

    counts: dict[str, int] = {}
    for detection in all_detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

    print(f"Total detections: {len(all_detections)}")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")
