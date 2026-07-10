"""Roboflow API calls and player filtering."""

import os
import base64
import requests
import numpy as np
import cv2

PLAYER_CLASSES = {"player", "goalkeeper"}
API_URL = "https://detect.roboflow.com/football-players-detection-3zvbc/1"


def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def detect_players(frame: np.ndarray, api_key: str) -> list[dict]:
    """Call Roboflow API and return filtered player detections.

    Returns list of dicts with keys: x, y, w, h, class, confidence.
    x, y are bounding box centre in pixel coordinates.
    """
    payload = encode_frame(frame)
    resp = requests.post(
        API_URL,
        params={"api_key": api_key},
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    detections = []
    for pred in data.get("predictions", []):
        if pred.get("class") in PLAYER_CLASSES:
            detections.append(
                {
                    "x": pred["x"],
                    "y": pred["y"],
                    "w": pred["width"],
                    "h": pred["height"],
                    "class": pred["class"],
                    "confidence": pred["confidence"],
                }
            )
    return detections
