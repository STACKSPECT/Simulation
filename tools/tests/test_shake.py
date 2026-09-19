import math

import numpy as np
import pytest

from stable_pallet.scenario import ShakeConfig, load_scenario
from stable_pallet.shake import (
    TRANSPORT_LEVELS_G,
    axis_unit,
    package_fell_off,
    pallet_tilt_deg,
    sine_jolt_force,
)
from stable_pallet.simulator import PalletizingSimulator


def _place_on_pallet(simulator: PalletizingSimulator, index: int, *, x_frac: float = 0.5, y_frac: float = 0.5) -> None:
    scenario = simulator.scenario
    package = scenario.packages[index]
    origin = scenario.simulation.pallet_origin
    center = (
        origin[0] + x_frac * scenario.pallet.width,
        origin[1] + y_frac * scenario.pallet.depth,
        scenario.simulation.pallet_height + package.size[2] / 2 + 0.002,
    )
    simulator._set_package_pose(index, center)


def _fast_shake(**overrides: object) -> ShakeConfig:
    values: dict = {
        "duration": 0.25,
        "hold_seconds": 0.02,
        "rest_seconds": 0.02,
        "settle_seconds": 0.35,
        "beam_settle_seconds": 1.2,
    }
    values.update(overrides)
    return ShakeConfig(**values)  # type: ignore[arg-type]


def test_default_protocol_is_five_transport_levels_on_three_axes() -> None:
    config = ShakeConfig()
    assert config.levels_g == TRANSPORT_LEVELS_G
    assert config.axes == ("x", "y", "z")
    assert len(config.levels_g) * len(config.axes) == 15
    assert config.levels_g[0] == pytest.approx(0.05)
    assert config.levels_g[-1] == pytest.approx(0.80)
    assert config.beam_width == pytest.approx(0.10)
    assert config.beam_height == pytest.approx(0.05)


def test_sine_jolt_pushes_then_brakes() -> None:
    direction = axis_unit("x")
    duration = 0.5
    peak = 100.0
    assert sine_jolt_force(0.0, duration, peak, direction)[0] == pytest.approx(0.0, abs=1e-9)
    assert sine_jolt_force(duration, duration, peak, direction)[0] == pytest.approx(0.0, abs=1e-9)
    assert sine_jolt_force(duration / 4, duration, peak, direction)[0] == pytest.approx(peak)
    assert sine_jolt_force(duration / 4, duration, peak, axis_unit("z"))[2] == pytest.approx(peak)


def test_a_package_outside_the_deck_has_fallen_off() -> None:
    assert not package_fell_off(np.array([0.0, 0.0, 0.20]), 1.2, 0.8, 0.144)
    assert package_fell_off(np.array([0.70, 0.0, 0.20]), 1.2, 0.8, 0.144)
    assert package_fell_off(np.array([0.0, 0.0, 0.02]), 1.2, 0.8, 0.144)


def test_the_pallet_stays_welded_until_the_jolt() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        assert simulator.data.eq_active[simulator.pallet_weld_id]
        _place_on_pallet(simulator, 0)
        simulator._step(0.25)
        origin = simulator.data.xpos[simulator.pallet_body_id].copy()
        simulator._step(0.25)
        assert np.allclose(simulator.data.xpos[simulator.pallet_body_id], origin, atol=1e-4)
    finally:
        simulator.close()


def test_a_gentle_jolt_moves_the_pallet_with_the_box() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        _place_on_pallet(simulator, 0)
        simulator._step(0.5)
        result = simulator.run_transport_trials(
            _fast_shake(levels_g=(0.15,), axes=("x",)),
            retract_arm=False,
        )
        trial = result["trials"][0]
        assert trial["axis_travel_mm"] > 5
        assert trial["packages"][0]["displacement_mm"] < 25
        assert not trial["packages"][0]["fell_off"]
        assert trial["held"]
        assert trial["score"] > 90
        assert result["summary"]["trial_count"] == 1
        assert result["summary"]["mean_score"] == trial["score"]
    finally:
        simulator.close()


def test_a_violent_jolt_shifts_the_load_on_the_deck() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        _place_on_pallet(simulator, 0, x_frac=0.22, y_frac=0.5)
        simulator._step(0.5)
        result = simulator.run_transport_trials(
            _fast_shake(levels_g=(2.8,), axes=("x",), max_shift_m=0.02),
            retract_arm=False,
        )
        package = result["trials"][0]["packages"][0]
        assert package["displacement_mm"] > 25 or package["fell_off"]
        assert not result["trials"][0]["held"]
        assert result["trials"][0]["score"] < 50
        assert result["trials"][0]["axis_travel_mm"] > 20
    finally:
        simulator.close()


def test_each_trial_rewinds_to_the_saved_stack() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        _place_on_pallet(simulator, 0, x_frac=0.22)
        simulator._step(0.5)
        result = simulator.run_transport_trials(
            _fast_shake(levels_g=(0.2, 2.5), axes=("x",)),
            retract_arm=False,
        )
        first = result["trials"][0]["packages"][0]["baseline_relative_center_m"]
        second = result["trials"][1]["packages"][0]["baseline_relative_center_m"]
        assert first == pytest.approx(second, abs=1e-4)
        assert result["trials"][1]["packages"][0]["displacement_mm"] > result["trials"][0]["packages"][0][
            "displacement_mm"
        ]
        restored = simulator._package_relative_to_pallet(0)[0]
        assert restored == pytest.approx(np.asarray(first), abs=5e-3)
    finally:
        simulator.close()


def test_instant_place_stacks_from_the_planner_without_the_arm() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        placed = simulator.place_from_plan()
        assert placed["placement_mode"] == "instant"
        assert placed["success"]
        assert len(simulator._placed_package_indices()) == len(simulator.scenario.packages)
        result = simulator.run_transport_trials(
            _fast_shake(levels_g=(0.15,), axes=("x", "y", "z")),
            retract_arm=False,
        )
        assert result["summary"]["trial_count"] == 3
        assert {trial["axis"] for trial in result["trials"]} == {"x", "y", "z"}
        baselines = [tuple(trial["packages"][0]["baseline_relative_center_m"]) for trial in result["trials"]]
        assert baselines[0] == pytest.approx(baselines[1], abs=1e-4)
        assert baselines[0] == pytest.approx(baselines[2], abs=1e-4)
    finally:
        simulator.close()


def test_pallet_tilt_is_the_deck_angle_about_the_beam() -> None:
    identity = np.eye(3)
    assert pallet_tilt_deg(identity, "x") == pytest.approx(0.0, abs=1e-9)
    assert pallet_tilt_deg(identity, "y") == pytest.approx(0.0, abs=1e-9)
    angle = math.radians(12.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    about_y = np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    assert pallet_tilt_deg(about_x, "x") == pytest.approx(-12.0)
    assert pallet_tilt_deg(about_y, "y") == pytest.approx(-12.0)


def test_a_centered_box_stays_up_on_the_narrow_beam() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        _place_on_pallet(simulator, 0)
        simulator._step(0.5)
        result = simulator.run_beam_trials(_fast_shake(), retract_arm=False)
        assert [trial["axis"] for trial in result["trials"]] == ["x", "y"]
        assert result["summary"]["trial_count"] == 2
        assert result["summary"]["held_all"]
        assert result["summary"]["by_axis"] == {"x": True, "y": True}
        for trial in result["trials"]:
            assert trial["seated"]
            assert abs(trial["tilt_deg"]) < 8
            assert not trial["packages"][0]["fell_off"]
    finally:
        simulator.close()


def test_an_offset_load_tips_across_the_narrow_beam() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        # Heavy parcel parked near the -X edge: CoM walks off a beam that runs along Y.
        _place_on_pallet(simulator, 4, x_frac=0.25, y_frac=0.5)
        simulator._step(0.5)
        result = simulator.run_beam_trials(_fast_shake(), retract_arm=False)
        by_axis = {trial["axis"]: trial for trial in result["trials"]}
        assert by_axis["x"]["held"]
        assert not by_axis["y"]["held"]
        assert abs(by_axis["y"]["tilt_deg"]) > 8 or not by_axis["y"]["seated"] or by_axis["y"]["packages"][0]["fell_off"]
    finally:
        simulator.close()


def test_the_beam_protocol_runs_x_then_y_from_the_same_stack() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"),
        simplified_graphics=True,
        measure_com=False,
    )
    try:
        _place_on_pallet(simulator, 0, x_frac=0.45, y_frac=0.52)
        simulator._step(0.5)
        result = simulator.run_beam_trials(_fast_shake(), retract_arm=False)
        first = result["trials"][0]["packages"][0]["baseline_relative_center_m"]
        second = result["trials"][1]["packages"][0]["baseline_relative_center_m"]
        assert first == pytest.approx(second, abs=1e-4)
        assert [trial["axis"] for trial in result["trials"]] == ["x", "y"]
        restored = simulator._package_relative_to_pallet(0)[0]
        assert restored == pytest.approx(np.asarray(first), abs=5e-3)
    finally:
        simulator.close()

