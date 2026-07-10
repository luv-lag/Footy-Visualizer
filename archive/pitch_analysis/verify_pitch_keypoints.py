#!/usr/bin/env python3
"""
Verification probe — DOES THE HOSTED PITCH KEYPOINT MODEL RETURN REAL PER-POINT
CONFIDENCES?  (gate before building the homography/grader module)

The grader keys entirely off per-keypoint confidence. If the hosted endpoint
returns coordinates only, or returns all-1.0 confidences, the grader design must
be rethought. This script makes that empirically visible and nothing else.

Usage:
    ROBOFLOW_API_KEY=<key> python verify_pitch_keypoints.py brasilscotland2.png
"""

from __future__ import annotations
import base64
import os
import sys
import json

import cv2
import numpy as np
import requests

PITCH_MODEL_ID = "football-field-detection-f07vi/14"
INFER_URL = "https://detect.roboflow.com"
TIMEOUT = 30


def _encode(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python verify_pitch_keypoints.py <image>")
    img_path = sys.argv[1]
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit("[ERROR] ROBOFLOW_API_KEY not set.")

    image = cv2.imread(img_path)
    if image is None:
        sys.exit(f"[ERROR] cannot read image: {img_path!r}")

    parts = PITCH_MODEL_ID.split("/")
    endpoint = f"{INFER_URL}/{parts[0]}/{parts[1]}"
    print(f"POST {endpoint}")
    resp = requests.post(
        endpoint,
        params={"api_key": key},
        data=_encode(image),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    preds = data.get("predictions", [])
    print(f"\nimage block : {data.get('image')}")
    print(f"top-level predictions: {len(preds)}")

    if not preds:
        print("\n[RESULT] No predictions returned. Top-level keys:", list(data.keys()))
        print(json.dumps(data, indent=2)[:1500])
        sys.exit("[STOP] Endpoint returned no predictions — verification inconclusive.")

    # Roboflow keypoint models nest per-point data under 'keypoints' on each prediction.
    first = preds[0]
    print(f"\nprediction[0] keys: {list(first.keys())}")
    kps = first.get("keypoints")
    if kps is None:
        print("\n[RESULT] No 'keypoints' field on prediction. This is NOT a keypoint")
        print("         response (looks like plain object detection). Full pred[0]:")
        print(json.dumps(first, indent=2)[:1500])
        sys.exit("[STOP] No per-point keypoints — grader design must be rethought.")

    confs = np.array([float(k.get("confidence", -1)) for k in kps], dtype=float)
    sample = kps[0]
    print(f"\nkeypoints on prediction[0]: {len(kps)}")
    print(f"sample keypoint[0]: {json.dumps(sample)}")
    print(f"\nconfidence stats over {len(confs)} points:")
    print(f"  has 'confidence' field : {'confidence' in sample}")
    print(f"  min  : {confs.min():.4f}")
    print(f"  max  : {confs.max():.4f}")
    print(f"  mean : {confs.mean():.4f}")
    print(f"  std  : {confs.std():.4f}")
    print(f"  #unique values : {len(np.unique(np.round(confs, 4)))}")
    print(f"  #points > 0.5  : {int((confs > 0.5).sum())}")
    print(f"  #points == 0   : {int((confs == 0).sum())}  (often = off-screen points)")

    # Verdict
    print("\n" + "=" * 60)
    if "confidence" not in sample or confs.max() < 0:
        print("VERDICT: NO per-point confidence field. STOP — rethink grader.")
    elif len(np.unique(np.round(confs, 4))) <= 1:
        print(f"VERDICT: confidences are CONSTANT ({confs[0]:.3f}). Useless for")
        print("         grading. STOP — rethink grader.")
    else:
        print("VERDICT: real, varying per-point confidences present. ✓")
        print("         Safe to build the grader keyed on these.")
    print("=" * 60)


if __name__ == "__main__":
    main()
