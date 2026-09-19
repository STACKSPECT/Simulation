from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from placing.metrics import METRIC_NAMES
from placing.types import CandidateBatch

DEFAULT_WEIGHTS: dict[str, float] = {
    "support_ratio": 1.0,
    "com_margin": 0.4,
    "lowness": 0.6,
    "void_fill": 1.2,
    "levelness": 0.8,
    "peak_penalty": 0.5,
    "lateral_proximity": 0.3,
    "edge_flush": 0.4,
    "seam_break": 0.2,
    "overhang": 0.5,
    "pallet_com": 0.0,
    "reachability": 0.7,
    "bridge": 0.5,
    "gap_waste": 0.8,
}


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    tie_eps: float = 1e-9
    proximity_band: float = 0.02
    levelness_scale: float = 0.2
    useful_gap: float = 0.0

    def weight_vector(self) -> dict[str, float]:
        return {name: float(self.weights.get(name, 0.0)) for name in METRIC_NAMES}


def weighted_scores(
    metrics: dict[str, np.ndarray],
    scoring: ScoringConfig | None = None,
) -> np.ndarray:
    """Input: per-metric arrays aligned with a CandidateBatch. Output: score in [0, 1].

    Normalised by the total weight. Every metric is already in [0, 1], so the
    weighted mean is too, which is what a caller can compare across boxes and
    across weight sets; the raw sum only means something next to its own weights.
    Dividing by a positive constant cannot reorder the candidates.
    """
    scoring = scoring or ScoringConfig()
    weights = scoring.weight_vector()
    n = next(iter(metrics.values())).shape[0] if metrics else 0
    score = np.zeros(n, dtype=np.float64)
    for name in METRIC_NAMES:
        score += weights[name] * metrics[name]
    total = sum(abs(w) for w in weights.values())
    return score / total if total > 0.0 else score


def pick_scored_index(
    batch: CandidateBatch,
    scores: np.ndarray,
    tie_eps: float = 1e-9,
) -> int | None:
    """Highest score among valid candidates; ties break by (z_base, y, x, yaw)."""
    valid_idx = np.flatnonzero(batch.valid)
    if valid_idx.size == 0:
        return None
    quantized = np.round(scores[valid_idx] / tie_eps)
    order = np.lexsort(
        (
            batch.yaw[valid_idx],
            batch.x[valid_idx],
            batch.y[valid_idx],
            batch.z_base[valid_idx],
            -quantized,
        )
    )
    return int(valid_idx[order[0]])
