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


class TeamColourModel:
    """The clip's two kit colours, fitted once on every outfield crop.

    One model, two consumers: :class:`TeamVoter` uses it to give each
    *track* a team (team assignment), and :class:`tracking.ColourGatedTracker`
    uses it to label each *detection* before association, so tracks never
    jump between kits (tracking cue).  Fitting it before tracking also keeps
    team 0 / team 1 consistent between both uses.
    """

    def __init__(self, max_fit_samples: int = 4000, seed: int = 0) -> None:
        self.max_fit_samples = max_fit_samples
        self.seed = seed
        self._km: KMeans | None = None
        self.separation_score: float = float("nan")

    @property
    def fitted(self) -> bool:
        return self._km is not None

    def fit(self, signatures: list[np.ndarray]) -> "TeamColourModel":
        X = np.stack(signatures) if signatures else np.zeros((0, 1))
        if len(X) < 2:
            return self
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_fit_samples:
            X = X[rng.choice(len(X), self.max_fit_samples, replace=False)]
        km = KMeans(n_clusters=2, random_state=self.seed, n_init=25).fit(X)
        if len(set(km.labels_.tolist())) == 2:
            self._km = km
            self.separation_score = float(silhouette_score(X, km.labels_))
        return self

    def predict(self, sig: np.ndarray | None) -> int:
        """Team 0/1 for one crop signature, or -1 (no signature / unfitted)."""
        if sig is None or self._km is None:
            return -1
        return int(self._km.predict(sig[None, :])[0])


class TeamVoter:
    """Per-track team vote over the crops of each track (team assignment)."""

    def __init__(self, model: TeamColourModel) -> None:
        self.model = model
        self._votes: dict[int, list[int]] = defaultdict(list)

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
        """Record one crop of a track (None is ignored)."""
        team = self.model.predict(sig)
        if team >= 0:
            self._votes[track_id].append(team)

    @property
    def separation_score(self) -> float:
        return self.model.separation_score

    def assign(self) -> dict[int, tuple[int, float]]:
        """Return ``{track_id: (team, vote_fraction)}`` for voted tracks."""
        out: dict[int, tuple[int, float]] = {}
        for t, votes in self._votes.items():
            v = np.array(votes)
            team = int(np.bincount(v, minlength=2).argmax())
            out[t] = (team, float((v == team).mean()))
        return out
