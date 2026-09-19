from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from random import Random

from .models import Package, Placement, StackState
from .pallet_com import compare_pallet_com, estimate_pallet_com
from .planner import (
    NoFeasiblePlacement,
    PlannerConfig,
    PlannerWeights,
    StablePalletPlanner,
)
from .scenario import Scenario
from .stability import validate_stack


@dataclass(frozen=True, slots=True)
class TrialMetric:
    method: str
    trial: int
    placed: int
    stable: bool
    height_m: float
    com_offset_m: float
    minimum_margin_m: float


def _run_sequence(
    scenario: Scenario, packages: list[Package], planner: StablePalletPlanner, method: str, trial: int
) -> TrialMetric:
    state = StackState(scenario.pallet)
    for index, _package in enumerate(packages):
        try:
            candidate = planner.plan_next(state, packages[index:])
        except NoFeasiblePlacement:
            break
        state = state.with_placement(candidate.placement)
    report = validate_stack(
        state,
        scenario.planner.minimum_support_ratio,
        scenario.planner.minimum_tipping_margin,
    )
    com_x, com_y, _ = state.global_com
    offset = hypot(com_x - state.pallet.width / 2, com_y - state.pallet.depth / 2)
    return TrialMetric(
        method,
        trial,
        len(state.placements),
        report.stable,
        state.max_height,
        offset,
        report.minimum_margin,
    )


def run_benchmark(scenario: Scenario, trials: int = 20, seed: int = 17) -> dict[str, object]:
    rng = Random(seed)
    stable = StablePalletPlanner(scenario.planner)
    first_fit = StablePalletPlanner(
        PlannerConfig(
            minimum_support_ratio=0.50,
            minimum_tipping_margin=0.0,
            beam_width=1,
            lookahead=1,
            candidates_per_node=48,
            heightmap_resolution=scenario.planner.heightmap_resolution,
            weights=PlannerWeights(0, 0, 0, 0, 0, 0),
        )
    )
    metrics: list[TrialMetric] = []
    for trial in range(trials):
        packages = list(scenario.packages)
        rng.shuffle(packages)
        metrics.append(_run_sequence(scenario, packages, first_fit, "first_fit", trial))
        metrics.append(_run_sequence(scenario, packages, stable, "stable_lookahead", trial))

    summary: dict[str, dict[str, float]] = {}
    for method in ("first_fit", "stable_lookahead"):
        selected = [metric for metric in metrics if metric.method == method]
        summary[method] = {
            "completion_rate": sum(metric.placed == len(scenario.packages) for metric in selected)
            / len(selected),
            "strict_stability_rate": sum(metric.stable for metric in selected) / len(selected),
            "mean_packages_placed": sum(metric.placed for metric in selected) / len(selected),
            "mean_height_m": sum(metric.height_m for metric in selected) / len(selected),
            "mean_com_offset_m": sum(metric.com_offset_m for metric in selected) / len(selected),
            "mean_minimum_margin_m": sum(metric.minimum_margin_m for metric in selected)
            / len(selected),
        }
    return {
        "scenario": scenario.name,
        "trials": trials,
        "seed": seed,
        "summary": summary,
        "raw": [asdict(metric) for metric in metrics],
    }


@dataclass(frozen=True, slots=True)
class ComTrialMetric:
    trial: int
    placed: int
    error_norm_mm: float
    error_xy_mm: float
    error_z_mm: float
    estimated_m: tuple[float, float, float]
    true_m: tuple[float, float, float]
    estimated_mass_kg: float
    true_mass_kg: float


def plan_stack(
    scenario: Scenario, packages: list[Package], planner: StablePalletPlanner
) -> list[Placement]:
    """Plan the whole sequence, stopping at the first package that will not fit."""
    state = StackState(scenario.pallet)
    for index, _package in enumerate(packages):
        try:
            candidate = planner.plan_next(state, packages[index:])
        except NoFeasiblePlacement:
            break
        state = state.with_placement(candidate.placement)
    return state.placements


def run_com_benchmark(scenario: Scenario, trials: int = 10, seed: int = 17) -> dict[str, object]:
    """Plan stacks, drop them onto the pallet, and measure CoM estimate error against MuJoCo."""
    from .simulator import PalletizingSimulator

    rng = Random(seed)
    planner = StablePalletPlanner(scenario.planner)
    simulator = PalletizingSimulator(scenario, measure_com=False, simplified_graphics=True)
    metrics: list[ComTrialMetric] = []
    try:
        for trial in range(trials):
            packages = list(scenario.packages)
            rng.shuffle(packages)
            planned = plan_stack(scenario, packages, planner)
            indices = simulator.apply_stack(planned)
            simulator._step(scenario.simulation.settle_seconds)
            comparison = compare_pallet_com(
                estimate_pallet_com(planned, scenario.pallet),
                simulator.measure_true_pallet_com(indices),
            )
            metrics.append(
                ComTrialMetric(
                    trial,
                    len(planned),
                    comparison.error_norm_m * 1_000,
                    comparison.error_xy_m * 1_000,
                    comparison.error_m[2] * 1_000,
                    comparison.estimated.com,
                    comparison.truth.com,
                    comparison.estimated.mass,
                    comparison.truth.mass,
                )
            )
    finally:
        simulator.close()

    summary = {
        "mean_error_norm_mm": sum(metric.error_norm_mm for metric in metrics) / len(metrics),
        "max_error_norm_mm": max(metric.error_norm_mm for metric in metrics),
        "mean_error_xy_mm": sum(metric.error_xy_mm for metric in metrics) / len(metrics),
        "max_error_xy_mm": max(metric.error_xy_mm for metric in metrics),
        "mean_abs_error_z_mm": sum(abs(metric.error_z_mm) for metric in metrics) / len(metrics),
        "mean_packages_placed": sum(metric.placed for metric in metrics) / len(metrics),
        "mean_mass_error_kg": sum(
            abs(metric.estimated_mass_kg - metric.true_mass_kg) for metric in metrics
        )
        / len(metrics),
    }
    return {
        "scenario": scenario.name,
        "trials": trials,
        "seed": seed,
        "summary": summary,
        "raw": [asdict(metric) for metric in metrics],
    }
