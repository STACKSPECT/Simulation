"""Contrato rápido del paletizado. No arranca MuJoCo ni usa la red."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from theker_telemetry import FAILURES  # noqa: E402

from scripts.palletize import oracle_for  # noqa: E402
from src import measure  # noqa: E402
from src.cell.render import VIEWS  # noqa: E402
from src.cell.scene import SOURCE_DECADE, SOURCES, Level, levels, load_configs  # noqa: E402
from src.contracts import Heightmap, PackageSpec, PlacementPlan  # noqa: E402
from src.episode import Episode  # noqa: E402
from src.planner.naive import BeamPlanner, GridPlanner  # noqa: E402
from src.telemetry import pallet_state_row, placement_row, run_config  # noqa: E402

CFG = load_configs(REPO)
EVENT_KINDS = {"perceive", "plan", "pick", "place", "settle", "fail"}
EXPECTED_FAILURES = {
    "no_detection", "ik_unreachable", "collision", "grasp_slip",
    "wrong_placement", "timeout", "stack_collapse", "overhang_violation",
}


def _placement(x=0.0, y=0.0, *, placed=True, cog=(0.0, 0.0, 0.0)):
    spec = PackageSpec("box-1", "std_m", (0.42, 0.30, 0.18), 4.2, np.array(cog))
    plan = PlacementPlan(np.array([x, y, 0.09]), 0.0, 1, "slot-1", 0.8, 1.0,
                         {"support": 1.0})
    box = SimpleNamespace(package_id="box-1", type_name="std_m")
    position = np.array([x, y, 0.09])
    return measure.Placement(
        box, spec, plan, position, 0.0, position.copy(), 0.0, 0.0,
        1.0, 0.0, placed,
    )


def test_rows_match_supabase_columns() -> None:
    placement = _placement()
    assert set(placement_row(0, placement)) == {
        "seq", "package_id", "package_type", "mass_kg", "dims_m", "layer",
        "planned_pose", "actual_pose", "error_xy_m", "error_yaw_rad",
        "support_ratio", "overhang_m", "placed",
    }
    state = measure.pallet_state([placement])
    assert set(pallet_state_row(0, state, 0.002)) == {
        "after_seq", "mass_kg", "cog_x", "cog_y", "cog_z",
        "stability_margin_m", "fill_ratio", "settle_drift_m",
    }


def test_event_rows_use_closed_vocabulary_and_unique_seq() -> None:
    episode = Episode(seed=1, n_objects=1)
    for seq, kind in enumerate(("perceive", "pick", "plan", "place", "settle", "fail")):
        episode.events.append({
            "ts": float(seq), "seq": seq, "kind": kind,
            "package_id": "box-1", "payload": {},
        })
    assert {event["kind"] for event in episode.events} <= EVENT_KINDS
    seqs = [event["seq"] for event in episode.events]
    assert seqs == sorted(set(seqs))


def test_views_and_failures_match_the_schema() -> None:
    assert set(CFG["cameras"]) == set(VIEWS) == {"top", "side", "iso", "camera"}
    assert set(FAILURES) == EXPECTED_FAILURES


def test_level_ids_encode_the_source() -> None:
    catalogue = levels(CFG)
    assert len(catalogue) == 9
    assert len(set(catalogue)) == 9
    assert {level.source for level in catalogue.values()} == set(SOURCES)
    for level in catalogue.values():
        assert level.id // 10 == SOURCE_DECADE[level.source]


def test_score_weights_sum_to_one() -> None:
    for name in ("heuristic", "planner"):
        assert abs(sum(CFG[name]["weights"].values()) - 1.0) < 1e-9


def test_oracle_is_any_stub_for_every_combination() -> None:
    for flags in itertools.product((False, True), repeat=3):
        assert oracle_for(*flags) is any(flags)


def test_planners_keep_their_contract() -> None:
    heightmap = Heightmap(np.zeros((20, 30)), (0.0, 0.0), 0.04)
    spec = PackageSpec("box-1", "std_m", (0.42, 0.30, 0.18), 4.2, np.zeros(3))
    naive = GridPlanner(CFG).choose(spec, heightmap)
    scored = BeamPlanner(CFG).choose(spec, heightmap)
    assert naive is not None and naive.score == 0.0 and naive.breakdown == {}
    assert scored is not None and 0.0 <= scored.score <= 1.0 and scored.breakdown
    impossible = PackageSpec("huge", "huge", (2.0, 2.0, 0.2), 1.0, np.zeros(3))
    assert GridPlanner(CFG).choose(impossible, heightmap) is None
    assert BeamPlanner(CFG).choose(impossible, heightmap) is None


def test_cog_counts_boxes_outside_tolerance() -> None:
    good = _placement(-0.05, placed=True)
    bad = _placement(0.05, placed=False, cog=(0.02, 0.0, 0.0))
    only_good = measure.pallet_state([good])
    both = measure.pallet_state([good, bad])
    assert both.mass_kg > only_good.mass_kg
    assert both.cog[0] > only_good.cog[0]


def test_a_box_left_on_the_table_is_not_on_the_pallet() -> None:
    scene = SimpleNamespace(pallet_dims=(1.2, 0.8))
    on = _placement(0.0, 0.0, placed=False)
    off = _placement(0.0, 0.7, placed=False)
    assert measure.on_pallet(scene, [on, off]) == [on]


def test_stability_margin_uses_the_support_polygon() -> None:
    base = [(-0.05, 0.0, 0.10, 0.06), (0.05, 0.0, 0.10, 0.06)]
    assert measure.support_polygon(base) == (-0.10, 0.10, -0.03, 0.03)
    centered = measure.stability_margin(0.0, 0.0, 0.05, base)
    assert round(centered, 9) == round(0.03 - 0.28 * 0.05, 9)
    assert measure.stability_margin(0.0, 0.0, 0.10, base) < centered
    assert measure.stability_margin(0.0, 0.05, 0.05, base) < 0.0
    assert measure.stability_margin(0.0, 0.0, 0.0, []) == 0.0


def test_run_config_names_source_level_and_real_pallet() -> None:
    scene = SimpleNamespace(
        cfg=CFG,
        level=Level(11, "table", "mesa", 1, ("std_m",)),
        boxes=[SimpleNamespace(
            package_id="box-1", type_name="std_m", dims_m=(0.42, 0.30, 0.18),
            mass_kg=4.2,
        )],
    )
    config = run_config(scene)
    assert config["pallet_size_m"] == [1.2, 0.8]
    assert config["source"] == "table" and config["level_name"] == "mesa"


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    measure.demo()
    print(f"{len(tests)} comprobaciones pasadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
