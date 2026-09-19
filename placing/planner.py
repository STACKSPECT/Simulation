from __future__ import annotations

from dataclasses import replace

import numpy as np

from placing.candidates import enumerate_candidates, placement_from_index
from placing.heightmap import HeightMap
from placing.metrics import METRIC_NAMES, MetricParams, compute_metrics
from placing.scoring import ScoringConfig, pick_scored_index, weighted_scores
from placing.support import footprint_features, resample_height_map
from placing.types import (
    BoxOrder,
    FeasibilityConfig,
    PalletState,
    Placement,
    PlanResult,
)


def _finite(height_map: HeightMap) -> HeightMap:
    """Drop NaN/inf cells to unobserved before anything reads the map.

    A depth camera returns no reading as NaN or inf, and one such cell is enough
    to poison the whole ranking in silence: it spreads through the summed-area
    tables behind `levelness` and the global max behind `peak_penalty`, so every
    candidate scores NaN, none of them is rejected, and the tie-break picks an
    arbitrary pose that the arm then goes and executes. "No valid reading" is
    exactly what `observed == False` means, so say that and let the existing
    filter deal with it.
    """
    bad = ~np.isfinite(height_map.heights)
    if not bad.any():
        return height_map
    clean = height_map.copy()
    clean.heights[bad] = 0.0
    clean.observed[bad] = False
    return clean


class PlacementPlanner:
    """Process 5: enumerate, filter, and score placements on a height map."""

    def __init__(
        self,
        feasibility: FeasibilityConfig | None = None,
        scoring: ScoringConfig | None = None,
        resolution: float | None = 0.01,
        robot_xy: tuple[float, float] | None = None,
        max_reach: float = 0.85,
    ) -> None:
        self.feasibility = feasibility or FeasibilityConfig()
        self.scoring = scoring or ScoringConfig()
        self.resolution = resolution
        self.robot_xy = robot_xy
        self.max_reach = max_reach

    def prepare_map(self, height_map: HeightMap) -> HeightMap:
        hmap = _finite(height_map)
        if self.resolution is None:
            return hmap
        return resample_height_map(hmap, self.resolution)

    def effective_feasibility(self) -> FeasibilityConfig:
        """Feasibility with the robot pose folded in, so reach is a hard filter."""
        return replace(
            self.feasibility,
            robot_xy=self.robot_xy if self.robot_xy is not None else self.feasibility.robot_xy,
            reach_max=self.max_reach if self.robot_xy is not None else self.feasibility.reach_max,
        )

    def plan(
        self,
        height_map: HeightMap,
        box: BoxOrder,
        pallet_state: PalletState | None = None,
    ) -> PlanResult:
        """Score every discrete pose of `box` on `height_map`.

        Input: pallet surface, inbound BoxOrder, optional running PalletState.
        Output: PlanResult with `best` set when at least one pose is feasible.
        """
        hmap = self.prepare_map(height_map)
        feasibility = self.effective_feasibility()
        # One sliding-window pass per footprint, shared by enumeration and scoring.
        features = footprint_features(hmap, box, feasibility)
        batch = enumerate_candidates(hmap, box, feasibility, features=features)
        params = MetricParams(
            proximity_band=self.scoring.proximity_band,
            levelness_scale=self.scoring.levelness_scale,
            max_stack_height=self.feasibility.max_stack_height,
            support_tol=self.feasibility.support_tol,
            clearance=self.feasibility.clearance,
            robot_xy=self.robot_xy,
            max_reach=self.max_reach,
            box_height=box.height,
            reach_knee=feasibility.reach_knee,
            reach_falloff=feasibility.reach_falloff,
            pallet_state=pallet_state,
            useful_gap=self.scoring.useful_gap,
        )
        metrics = compute_metrics(
            hmap, box, batch, params=params, cfg=feasibility, features=features
        )
        scores = weighted_scores(metrics, self.scoring)
        if len(batch):
            scores = np.where(batch.valid, scores, -np.inf)
        index = pick_scored_index(batch, scores, self.scoring.tie_eps)
        best: Placement | None = None
        if index is not None:
            best = placement_from_index(batch, box, index)
            best = replace(
                best,
                metrics={name: float(metrics[name][index]) for name in METRIC_NAMES},
                score=float(scores[index]),
            )
        return PlanResult(
            best=best,
            batch=batch,
            scores=scores,
            metrics=metrics,
            n_valid=batch.n_valid,
            height_map=hmap,
        )


def ranked_placements(
    result: PlanResult,
    box: BoxOrder,
    limit: int = 12,
    min_separation: float = 0.09,
) -> list[Placement]:
    """Best valid placements in score order, thinned so retries are distinct.

    Neighbouring cells score almost identically, so an unfiltered ranking would
    retry the same pose a centimetre over and fail the same way. Spacing the
    retries out also spreads them across the arm's workspace, where whole
    regions can be unreachable for reasons the height map cannot see.
    """
    valid = np.flatnonzero(result.batch.valid)
    if valid.size == 0:
        return []
    order = valid[np.argsort(-result.scores[valid], kind="stable")]
    chosen: list[Placement] = []
    for index in order:
        x, y = float(result.batch.x[index]), float(result.batch.y[index])
        if any(np.hypot(p.x - x, p.y - y) < min_separation for p in chosen):
            continue
        placement = placement_from_index(result.batch, box, int(index))
        chosen.append(
            replace(
                placement,
                metrics={name: float(result.metrics[name][index]) for name in METRIC_NAMES},
                score=float(result.scores[index]),
            )
        )
        if len(chosen) >= limit:
            break
    return chosen


def planner_from_config(planning) -> PlacementPlanner:
    """Build a planner from `PlanningConfig` / `configs/default.yaml`."""
    feasibility = FeasibilityConfig(
        clearance=planning.clearance,
        support_tol=planning.support_tol,
        allow_unobserved=planning.allow_unobserved,
        min_support_ratio=planning.min_support_ratio,
        max_stack_height=planning.max_stack_height,
        yaws=planning.yaws,
        reach_min=float(getattr(planning, "reach_min", 0.0)),
        reach_knee=float(getattr(planning, "reach_knee", 0.25)),
        reach_falloff=float(getattr(planning, "reach_falloff", 0.65)),
    )
    scoring = ScoringConfig(
        weights=dict(planning.weights),
        proximity_band=planning.proximity_band,
        levelness_scale=planning.levelness_scale,
        useful_gap=float(getattr(planning, "useful_gap", 0.0)),
    )
    robot_xy = None
    if getattr(planning, "robot_x", None) is not None and getattr(planning, "robot_y", None) is not None:
        robot_xy = (float(planning.robot_x), float(planning.robot_y))
    return PlacementPlanner(
        feasibility,
        scoring,
        resolution=planning.resolution,
        robot_xy=robot_xy,
        max_reach=float(getattr(planning, "max_reach", 0.85)),
    )
