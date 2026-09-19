import numpy as np
import pytest

from stable_pallet.benchmark import run_com_benchmark
from stable_pallet.cli import build_parser
from stable_pallet.models import Package, Pallet, Placement
from stable_pallet.pallet_com import compare_pallet_com, estimate_pallet_com
from stable_pallet.scenario import Scenario, SimulationConfig
from stable_pallet.simulator import PalletizingSimulator


def _box(name: str, size=(0.4, 0.4, 0.2), mass=5.0, com=(0.0, 0.0, 0.0)) -> Package:
    return Package(name, size, mass, com)


def _two_box_scenario() -> Scenario:
    return Scenario(
        "two-box-com",
        Pallet(1.2, 0.8),
        (
            _box("base", (0.40, 0.40, 0.20), 5.0, (0.05, -0.02, -0.01)),
            _box("top", (0.30, 0.28, 0.16), 3.0, (-0.04, 0.03, 0.02)),
        ),
        simulation=SimulationConfig(settle_seconds=0.4, pallet_origin=(0.0, -0.40), pallet_height=0.144),
    )


def test_empty_stack_sits_at_the_pallet_centre() -> None:
    pallet = Pallet(1.2, 0.8)
    estimate = estimate_pallet_com([], pallet)
    assert estimate.mass == 0.0
    assert estimate.com == (0.6, 0.4, 0.0)


def test_two_packages_give_the_mass_weighted_average() -> None:
    pallet = Pallet(1.2, 0.8)
    base = Placement(_box("base", mass=8.0, com=(0.05, 0.0, -0.01)), 0.10, 0.10, 0.0, 0)
    top = Placement(_box("top", size=(0.3, 0.3, 0.2), mass=2.0, com=(0.0, 0.04, 0.02)), 0.15, 0.15, 0.20, 90)
    estimate = estimate_pallet_com([base, top], pallet)
    expected = tuple(
        (8.0 * base_c + 2.0 * top_c) / 10.0
        for base_c, top_c in zip(base.com_world, top.com_world, strict=True)
    )
    assert estimate.mass == pytest.approx(10.0)
    assert estimate.com == pytest.approx(expected)


def test_yaw_ninety_rotates_the_package_centre_of_mass() -> None:
    package = _box("turned", com=(0.06, 0.02, 0.0))
    placement = Placement(package, 0.2, 0.1, 0.0, 90)
    cx, cy, cz = placement.center
    assert placement.com_world == pytest.approx((cx - 0.02, cy + 0.06, cz))


def test_comparison_reports_the_vector_from_truth_to_the_estimate() -> None:
    pallet = Pallet(1.2, 0.8)
    estimated = estimate_pallet_com(
        [Placement(_box("only", mass=4.0, com=(0.03, -0.01, 0.0)), 0.2, 0.1, 0.0)],
        pallet,
    )
    truth = estimate_pallet_com(
        [Placement(_box("only", mass=4.0, com=(0.0, 0.0, 0.0)), 0.2, 0.1, 0.0)],
        pallet,
    )
    comparison = compare_pallet_com(estimated, truth)
    assert comparison.error_m == pytest.approx((0.03, -0.01, 0.0))
    assert comparison.error_xy_m == pytest.approx(np.hypot(0.03, 0.01))


def test_estimate_matches_mujoco_before_anything_moves() -> None:
    scenario = _two_box_scenario()
    placements = [
        Placement(scenario.packages[0], 0.40, 0.20, 0.0, 0),
        Placement(scenario.packages[1], 0.45, 0.24, 0.20, 90),
    ]
    simulator = PalletizingSimulator(scenario, measure_com=False, simplified_graphics=True)
    try:
        indices = simulator.apply_stack(placements)
        estimated = estimate_pallet_com(placements, scenario.pallet)
        truth = simulator.measure_true_pallet_com(indices)
        assert truth.mass == pytest.approx(estimated.mass, abs=1e-9)
        assert np.allclose(truth.com, estimated.com, atol=1e-9)
    finally:
        simulator.close()


def test_settling_leaves_a_small_error_against_the_commanded_stack() -> None:
    scenario = _two_box_scenario()
    placements = [
        Placement(scenario.packages[0], 0.40, 0.20, 0.0, 0),
        Placement(scenario.packages[1], 0.45, 0.24, 0.20, 90),
    ]
    simulator = PalletizingSimulator(scenario, measure_com=False, simplified_graphics=True)
    try:
        indices = simulator.apply_stack(placements, lift=0.001)
        simulator._step(scenario.simulation.settle_seconds)
        comparison = compare_pallet_com(
            estimate_pallet_com(placements, scenario.pallet),
            simulator.measure_true_pallet_com(indices),
        )
        assert comparison.error_xy_m < 0.02
        assert abs(comparison.error_m[2]) < 0.03
    finally:
        simulator.close()


def test_cli_exposes_the_pallet_com_benchmark() -> None:
    args = build_parser().parse_args(["benchmark-com", "--trials", "3", "--seed", "9"])
    assert args.command == "benchmark-com"
    assert args.trials == 3
    assert args.seed == 9
    assert args.output == "artifacts/pallet_com_benchmark.json"


def test_com_benchmark_runs_against_mujoco() -> None:
    result = run_com_benchmark(_two_box_scenario(), trials=1, seed=0)
    assert result["raw"][0]["placed"] == 2
    assert result["summary"]["mean_mass_error_kg"] == 0.0
    assert result["summary"]["mean_error_norm_mm"] < 20.0
