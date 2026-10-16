"""teams.py -- team assignment per *track* instead of per frame.

The single-frame pipeline clusters each frame's ~20 torso crops on its own,
so team labels can flicker frame to frame (and cluster ids can swap between
frames).  For video, ``team_id_v2``'s colour signatures are pooled across
the whole clip, clustered ONCE (so team 0 / team 1 mean the same thing in
every frame), and each track takes the majority vote of its own crops.

Shadows: ``team_id_v2``'s signature keeps brightness information (dark /
light hue bands, lightness bins), so on a pitch half in hard shadow it
clusters "sunlit vs shaded" instead of "team A vs team B" -- 50 % purity on
SoccerNet GSR clip SNGS-048.  :func:`robust_signature` concatenates two
illumination-tolerant views of the same crop:

* the v2 signature with brightness collapsed (hue mass + one achromatic
  bin), and
* the v2 signature of the crop after gray-world white balance and
  brightness normalisation (removes the shade's colour cast and darkening).

Measured on two GSR clips (per crop / per track vote): SNGS-048 50 % / 55 %
-> 97 % / 100 %; SNGS-082 99 % / 100 % -> 99 % / 100 %.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from detection import Detection
import cv2

from team_id_v2 import (
    MIN_DETECTION_CONFIDENCE,
    SIGNATURE_HUE_BINS,
    compute_signature,
    extract_torso_crop,
)


def _hue_mass(sig: np.ndarray) -> np.ndarray:
    """v2 signature with dark/light bands merged and lightness collapsed."""
    p = sig**2  # undo the Hellinger mapping
    hue = p[: 2 * SIGNATURE_HUE_BINS].reshape(-1, 2).sum(axis=1)
    v = np.append(hue, p[2 * SIGNATURE_HUE_BINS :].sum())
    total = v.sum()
    return np.sqrt(v / total) if total > 0 else v


def _balanced(crop: np.ndarray) -> np.ndarray:
    """Gray-world white balance, then scale brightness to a fixed level."""
    f = crop.astype(np.float64)
    mean = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f = np.clip(f * (mean.mean() / mean), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[..., 2] = np.clip(hsv[..., 2] * (200.0 / (np.percentile(hsv[..., 2], 90) + 1e-6)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def robust_signature(crop: np.ndarray) -> np.ndarray:
    """Illumination-tolerant team signature (see module docstring)."""
    return np.concatenate([_hue_mass(compute_signature(crop)), compute_signature(_balanced(crop))])


class TeamVoter:
    """Collect torso signatures per track, then cluster once and vote."""

    def __init__(self, max_fit_samples: int = 4000, seed: int = 0) -> None:
        self.max_fit_samples = max_fit_samples
        self.seed = seed
        self._sigs: dict[int, list[np.ndarray]] = defaultdict(list)
        self.separation_score: float = float("nan")

    @staticmethod
    def signature(frame: np.ndarray, det: Detection) -> np.ndarray | None:
        """Torso colour signature of one detection (outfield players only, as in v2)."""
        if det.class_name != "player" or det.confidence < MIN_DETECTION_CONFIDENCE:
            return None
        try:
            sig = robust_signature(extract_torso_crop(frame, det))
        except ValueError:
            return None
        return sig if sig.any() else None

    def add(self, track_id: int, sig: np.ndarray | None) -> None:
        """Record one crop signature for a track (None is ignored)."""
        if sig is not None:
            self._sigs[track_id].append(sig)

    def assign(self) -> dict[int, tuple[int, float]]:
        """Return ``{track_id: (team, vote_fraction)}`` for voted tracks."""
        ids = [t for t, s in self._sigs.items() for _ in s]
        if len(set(ids)) < 2:
            return {}
        X = np.stack([s for t in self._sigs for s in self._sigs[t]])
        ids = np.array(ids)
        rng = np.random.default_rng(self.seed)
        fit_idx = (
            rng.choice(len(X), self.max_fit_samples, replace=False)
            if len(X) > self.max_fit_samples
            else np.arange(len(X))
        )
        km = KMeans(n_clusters=2, random_state=self.seed, n_init=25).fit(X[fit_idx])
        labels = km.predict(X)
        if len(set(km.labels_.tolist())) == 2:
            self.separation_score = float(silhouette_score(X[fit_idx], km.labels_))
        out: dict[int, tuple[int, float]] = {}
        for t in self._sigs:
            votes = labels[ids == t]
            team = int(np.bincount(votes, minlength=2).argmax())
            out[t] = (team, float((votes == team).mean()))
        return out
