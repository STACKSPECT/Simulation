from stable_pallet.models import Package, Pallet, StackState
from stable_pallet.planner import PlannerConfig, StablePalletPlanner
from stable_pallet.scenario import load_scenario
from stable_pallet.stability import overlaps_3d, validate_stack


def test_first_package_is_centered() -> None:
    pallet = Pallet(1.2, 1.0)
    package = Package("first", (0.4, 0.3, 0.2), 5.0)
    planner = StablePalletPlanner(PlannerConfig(lookahead=1))
    candidate = planner.plan_next(StackState(pallet), [package])
    center_x, center_y, _ = candidate.placement.center
    assert abs(center_x - pallet.width / 2) < 1e-8
    assert abs(center_y - pallet.depth / 2) < 1e-8


def test_demo_scenario_produces_valid_non_overlapping_stack() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    planner = StablePalletPlanner(scenario.planner)
    state = StackState(scenario.pallet)
    for index, _package in enumerate(scenario.packages):
        candidate = planner.plan_next(state, list(scenario.packages[index:]))
        state = state.with_placement(candidate.placement)
    report = validate_stack(
        state,
        scenario.planner.minimum_support_ratio,
        scenario.planner.minimum_tipping_margin,
    )
    assert report.stable, report.reason
    assert all(
        not overlaps_3d(first, second)
        for index, first in enumerate(state.placements)
        for second in state.placements[index + 1 :]
    )
