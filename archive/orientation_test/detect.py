import base64
import os

import requests

ENDPOINT = "https://detect.roboflow.com/football-players-detection-3zvbc/1"
TARGET_CLASSES = {"player", "goalkeeper"}


def detect_players(image_path: str) -> list[tuple[float, float]]:
    api_key = os.environ["ROBOFLOW_API_KEY"]

    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("utf-8")

    resp = requests.post(
        ENDPOINT,
        params={"api_key": api_key},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    return [
        (p["x"], p["y"])
        for p in data.get("predictions", [])
        if p.get("class") in TARGET_CLASSES
    ]
