"""Placement heuristic: enumerate every discrete pose, filter, score, pick one.

numpy and nothing else. No simulator, no cameras, no config loader: what crosses
the boundary in is a `HeightMap` and a `BoxOrder`, and what comes back out is a
`Placement` with its score broken down term by term. Keeping it that way is what
lets the heuristic be tested, and tuned, without starting a simulator.

`python -m placing` runs the self-check.
"""

from __future__ import annotations

from placing.candidates import (
    best_feasible,
    enumerate_candidates,
    pick_best_index,
    reject_summary,
)
from placing.heightmap import HeightMap, empty_height_map
from placing.metrics import METRIC_NAMES, compute_metrics
from placing.planner import PlacementPlanner, planner_from_config, ranked_placements
from placing.scoring import DEFAULT_WEIGHTS, ScoringConfig, weighted_scores
from placing.support import footprint_features, resample_height_map, yaw_set
from placing.types import (
    BoxOrder,
    CandidateBatch,
    FeasibilityConfig,
    PalletState,
    Placement,
    PlanResult,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "METRIC_NAMES",
    "BoxOrder",
    "CandidateBatch",
    "FeasibilityConfig",
    "HeightMap",
    "PalletState",
    "Placement",
    "PlacementPlanner",
    "PlanResult",
    "ScoringConfig",
    "best_feasible",
    "choose",
    "compute_metrics",
    "empty_height_map",
    "enumerate_candidates",
    "footprint_features",
    "pick_best_index",
    "plan",
    "planner_from_config",
    "ranked_placements",
    "reject_summary",
    "resample_height_map",
    "weighted_scores",
    "yaw_set",
]


def plan(
    height_map: HeightMap,
    box: BoxOrder,
    pallet_state: PalletState | None = None,
    planner: PlacementPlanner | None = None,
) -> PlanResult:
    """Score every discrete pose of `box` on `height_map`.

    `result.best` is None when nothing is feasible; `reject_summary(result.batch)`
    then says which constraint did the rejecting.
    """
    planner = planner or PlacementPlanner()
    return planner.plan(height_map, box, pallet_state=pallet_state)


def choose(
    height_map: HeightMap,
    box: BoxOrder,
    pallet_state: PalletState | None = None,
    planner: PlacementPlanner | None = None,
) -> Placement | None:
    """The chosen pose, or None when the box does not fit anywhere.

    None is a normal outcome, not an error: the caller decides what to do with a
    box that does not fit and records it. Raising from here would take the whole
    run down instead.
    """
    return plan(height_map, box, pallet_state, planner).best
