from pathlib import Path

import pytest

from stable_pallet.cli import build_parser
from stable_pallet.experiments import (
    DEFAULT_SCENARIO,
    EXPERIMENTS,
    Experiment,
    RunOptions,
    _from_source,
    _measures_com,
    experiment,
    run_experiment,
    summarise,
)
from stable_pallet.scenario import load_scenario


def test_every_experiment_has_its_own_name_and_artifact() -> None:
    keys = [item.key for item in EXPERIMENTS]
    outputs = [item.output for item in EXPERIMENTS]
    assert len(set(keys)) == len(keys)
    assert len(set(outputs)) == len(outputs)


def test_the_prepared_list_covers_the_demo_and_the_numbers() -> None:
    keys = {item.key for item in EXPERIMENTS}
    assert {"eval", "pallet-test", "palletize", "transport"} <= keys
    assert experiment("pallet-test").instant_place
    assert experiment("eval").shake
    assert experiment("palletize").uses_robot


def test_the_trailer_runs_are_offered_beside_the_conveyor_ones() -> None:
    """Unloading a trailer is its own experiment, not a flag on the conveyor ones."""
    assert experiment("truck-unload").source == "truck"
    assert experiment("truck-eval").source == "truck"
    assert {item.key for item in EXPERIMENTS if item.source == "conveyor"} >= {"eval", "palletize"}


def test_the_source_decides_whether_the_cell_has_a_trailer() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO)
    assert scenario.simulation.truck is None

    assert _from_source(scenario, "conveyor") is scenario
    assert _from_source(scenario, "truck").simulation.truck is not None
    with pytest.raises(ValueError, match="belt"):
        _from_source(scenario, "belt")


def test_a_summary_reports_what_the_trailer_held_and_how_still_it_stayed() -> None:
    item = experiment("truck-unload")
    result = {
        "placed": 8,
        "expected": 8,
        "success": True,
        "state": {"max_height": 0.51},
        "minimum_support_ratio": 0.68,
        "minimum_tipping_margin_m": 0.028,
        "truck": {
            "slots": [{}] * 8,
            "columns": 3,
            "tallest_column_m": 0.62,
            "followed_plan": True,
            "max_load_disturbance_mm": 0.4,
        },
    }

    lines = "\n".join(summarise(item, result))

    assert "8 cajas en 3 columnas" in lines
    assert "620 mm" in lines
    assert "0.4 mm" in lines


def test_teleporting_the_cartons_rules_out_weighing_them() -> None:
    """Nothing is picked up, so there is no wrist reading to take."""
    item = experiment("pallet-test")
    assert not item.uses_robot
    assert not _measures_com(item, RunOptions(measure_com=True))


def test_only_the_cell_experiments_can_be_watched() -> None:
    for item in EXPERIMENTS:
        assert item.watchable == (item.kind in {"cell", "generated"})
        if item.instant_place:
            assert not item.uses_robot


def test_the_command_line_offers_exactly_the_prepared_list() -> None:
    parser = build_parser()
    for item in EXPERIMENTS:
        args = parser.parse_args(["run", item.key])
        assert args.experiment == item.key
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "not-an-experiment"])


def test_asking_for_an_unknown_experiment_says_what_there_is() -> None:
    with pytest.raises(KeyError, match="pallet-test"):
        experiment("palet-test")


def test_running_the_planner_experiment_writes_its_artifact(tmp_path: Path) -> None:
    item = experiment("plan")
    output = tmp_path / "plan.json"

    result = run_experiment(item, RunOptions(output=output))

    assert output.exists()
    assert result["experiment"] == "plan"
    assert result["placed"] == result["expected"]
    assert result["success"]


def test_a_summary_reads_the_result_without_knowing_what_ran() -> None:
    item = experiment("transport")
    result = {
        "placed": 6,
        "expected": 6,
        "success": True,
        "state": {"max_height": 0.44},
        "minimum_support_ratio": 0.71,
        "minimum_tipping_margin_m": 0.031,
        "com_measurements": [{"error_mm": 8.0, "replanned": True}, {"error_mm": 4.0, "replanned": False}],
        "pallet_com": {"error_norm_mm": 12.5, "error_xy_mm": 9.0},
        "shake": {"summary": {"mean_score": 88.1, "min_score": 61.0, "held_all": True}},
        "beam": {"summary": {"by_axis": {"x": True, "y": False}}},
    }

    lines = "\n".join(summarise(item, result))

    assert "6/6" in lines
    assert "6.0 mm" in lines  # mean of the two wrist readings
    assert "1 replanificadas" in lines
    assert "88.1" in lines
    assert "y vuelca" in lines


def test_an_unknown_kind_is_rejected_rather_than_silently_skipped() -> None:
    broken = Experiment(key="x", title="x", description="x", kind="nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        run_experiment(broken)
