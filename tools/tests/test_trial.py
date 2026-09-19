import pytest

from stable_pallet.cli import build_parser
from stable_pallet.generator import generate_boxes
from stable_pallet.shake import score_trial
from stable_pallet.trial import EVAL_BOX_COUNT, EVAL_SEED, build_generated_scenario, run_generated_eval
from stable_pallet.truck import plan_truck_load, unload_order


def _trial(*, displacement_mm: float, fell_off: bool = False, held: bool | None = None) -> dict:
    shift_m = displacement_mm / 1_000.0
    if held is None:
        held = (not fell_off) and shift_m <= 0.030
    return {
        "held": held,
        "packages": [
            {
                "package_id": "BOX-01",
                "displacement_mm": displacement_mm,
                "fell_off": fell_off,
            }
        ],
    }


def test_score_is_100_when_the_load_does_not_move() -> None:
    assert score_trial(_trial(displacement_mm=0.0), 0.030) == 100.0


def test_score_is_50_at_the_hold_threshold() -> None:
    assert score_trial(_trial(displacement_mm=30.0), 0.030) == 50.0


def test_score_is_zero_when_a_box_falls_off() -> None:
    assert score_trial(_trial(displacement_mm=5.0, fell_off=True), 0.030) == 0.0


def test_score_is_zero_beyond_twice_the_hold_threshold() -> None:
    assert score_trial(_trial(displacement_mm=60.0), 0.030) == 0.0
    assert score_trial(_trial(displacement_mm=90.0), 0.030) == 0.0


def test_eval_defaults_are_six_boxes_arriving_in_a_trailer() -> None:
    scenario = build_generated_scenario()
    assert EVAL_BOX_COUNT == 6
    assert len(scenario.packages) == 6
    assert scenario.pallet.width == 1.2
    assert scenario.pallet.depth == 0.8
    assert scenario.packages == generate_boxes(6, seed=EVAL_SEED)
    # Same parcels, same count as the conveyor eval; only where they arrive changed.
    assert scenario.simulation.truck is not None
    assert len(plan_truck_load(scenario.packages, scenario.simulation.truck, seed=EVAL_SEED)) == 6
    assert build_generated_scenario(source="conveyor").simulation.truck is None


def test_the_eval_command_is_wired_on_the_cli() -> None:
    args = build_parser().parse_args(["eval", "--no-measure-com", "--count", "6"])
    assert args.command == "eval"
    assert args.count == 6
    assert args.seed == EVAL_SEED
    assert args.measure_com is False
    assert args.output == "artifacts/eval.json"
    assert args.source is None  # the trailer, unless the command line says otherwise
    assert build_parser().parse_args(["eval", "--source", "conveyor"]).source == "conveyor"


@pytest.fixture(scope="module")
def generated_eval() -> dict:
    """Robot picks six random cartons, the stack is frozen, then 15 transport jolts are scored."""
    return run_generated_eval(
        count=EVAL_BOX_COUNT,
        seed=EVAL_SEED,
        measure_com=True,
        simplified_graphics=True,
    )


def test_the_cell_empties_the_trailer_from_the_top_down(generated_eval: dict) -> None:
    """The order is decided from the measured load, so it has to match the only legal one."""
    truck = generated_eval["truck"]
    scenario = build_generated_scenario(EVAL_BOX_COUNT, EVAL_SEED)
    expected = [
        slot.package_id
        for slot in unload_order(
            plan_truck_load(scenario.packages, scenario.simulation.truck, seed=EVAL_SEED)
        )
    ]

    assert generated_eval["source"] == "truck"
    assert truck["pick_order"] == expected
    assert truck["followed_plan"]
    assert truck["columns"] >= 2
    assert truck["tallest_column_m"] > 0.30


def test_taking_the_top_carton_leaves_the_rest_of_the_load_standing(generated_eval: dict) -> None:
    """A pick that drags its neighbours out with it is not an unload."""
    assert generated_eval["truck"]["max_load_disturbance_mm"] < 5.0


def test_robot_picks_and_places_every_generated_box(generated_eval: dict) -> None:
    assert generated_eval["placement_mode"] == "robot"
    assert generated_eval["placed"] == EVAL_BOX_COUNT
    assert generated_eval["success"]
    assert len(generated_eval["measurements"]) == EVAL_BOX_COUNT
    assert len(generated_eval["com_measurements"]) == EVAL_BOX_COUNT
    assert generated_eval["state"]["max_height"] < generated_eval["state"]["pallet"]["max_height"]
    assert generated_eval["state"]["max_height"] > 0.20


def test_interpolated_wrist_probe_keeps_com_accuracy(generated_eval: dict) -> None:
    declared = {package["id"]: package for package in generated_eval["generator"]["packages"]}
    for item in generated_eval["com_measurements"]:
        truth = declared[item["package_id"]]
        assert item["trustworthy"], item
        assert item["error_mm"] < 1.0
        assert item["measured_mass_kg"] == pytest.approx(truth["mass"], abs=0.02)


def test_saved_stack_is_replayed_for_every_shake_trial(generated_eval: dict) -> None:
    trials = generated_eval["shake"]["trials"]
    assert len(trials) == 15
    first = trials[0]["packages"][0]["baseline_relative_center_m"]
    for trial in trials[1:]:
        assert trial["packages"][0]["baseline_relative_center_m"] == pytest.approx(first, abs=1e-4)


def test_each_shake_trial_reports_a_score(generated_eval: dict) -> None:
    shake = generated_eval["shake"]
    trials = shake["trials"]
    scores = [trial["score"] for trial in trials]
    assert len(scores) == 15
    assert all(0.0 <= score <= 100.0 for score in scores)
    summary = shake["summary"]
    assert summary["mean_score"] == pytest.approx(sum(scores) / len(scores), abs=0.05)
    assert summary["min_score"] == min(scores)
    assert 0.0 <= summary["weighted_score"] <= 100.0

    by_axis = {(trial["peak_accel_g"], trial["axis"]): trial for trial in trials}
    gentle = by_axis[(0.05, "x")]["score"]
    hard = by_axis[(0.80, "x")]["score"]
    assert gentle > 90
    assert hard < gentle
