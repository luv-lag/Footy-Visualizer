"""teams.py -- team assignment per *track* instead of per frame.

The single-frame pipeline clusters each frame's ~20 torso crops on its own,
so team labels can flicker frame to frame (and cluster ids can swap between
frames).  For video, ``team_id_v2``'s colour signatures are pooled across
the whole clip, clustered ONCE (so team 0 / team 1 mean the same thing in
every frame), and each track takes the majority vote of its own crops.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from detection import Detection
from team_id_v2 import MIN_DETECTION_CONFIDENCE, compute_signature, extract_torso_crop


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
            sig = compute_signature(extract_torso_crop(frame, det))
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
