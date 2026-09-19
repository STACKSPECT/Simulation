import math

import pytest

from stable_pallet.generator import generate_boxes
from stable_pallet.models import Package
from stable_pallet.trial import EVAL_BOX_COUNT, EVAL_SEED
from stable_pallet.truck import (
    TruckBay,
    TruckLoadTooBig,
    TruckSlot,
    load_summary,
    plan_truck_load,
    unload_order,
)

_SEEDS = range(60)


def _load(seed: int, count: int = EVAL_BOX_COUNT, bay: TruckBay | None = None):
    packages = generate_boxes(count, seed=seed)
    return packages, plan_truck_load(packages, bay or TruckBay(), seed=seed)


def _corners(slot: TruckSlot) -> tuple[float, float, float, float]:
    """Footprint of a slot on the deck, widened by how crooked it was stacked.

    `footprint` is already the carton turned to 0 or 90 degrees; what is left over is
    the few degrees of slop the loader left behind.
    """
    angle = math.radians(slot.yaw - 90 * round(slot.yaw / 90))
    width, depth = slot.footprint
    spread_x = abs(width * math.cos(angle)) + abs(depth * math.sin(angle))
    spread_y = abs(width * math.sin(angle)) + abs(depth * math.cos(angle))
    return (
        slot.center[0] - spread_x / 2,
        slot.center[0] + spread_x / 2,
        slot.center[1] - spread_y / 2,
        slot.center[1] + spread_y / 2,
    )


def test_every_carton_stays_inside_the_bay_the_arm_can_reach() -> None:
    bay = TruckBay()
    min_x, max_x, min_y, max_y = bay.bounds
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        for slot in slots:
            left, right, near, far = _corners(slot)
            # The jitter of a hand-stacked load is allowed to hang a few mm over the
            # planned edge; the bay itself already sits inside the reachable envelope.
            assert left > min_x - 0.02, (seed, slot.package_id)
            assert right < max_x + 0.02, (seed, slot.package_id)
            assert near > min_y - 0.02, (seed, slot.package_id)
            assert far < max_y + 0.02, (seed, slot.package_id)
            assert slot.top <= bay.ceiling + 1e-9, (seed, slot.package_id)


def test_cartons_never_occupy_the_same_space() -> None:
    """Two cartons that overlap in plan have to be stacked, not interleaved."""
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        for first in range(len(slots)):
            for second in range(first + 1, len(slots)):
                one, other = slots[first], slots[second]
                l1, r1, n1, f1 = _corners(one)
                l2, r2, n2, f2 = _corners(other)
                overlap_xy = min(r1, r2) - max(l1, l2) > 1e-6 and min(f1, f2) - max(n1, n2) > 1e-6
                overlap_z = min(one.top, other.top) - max(one.bottom, other.bottom) > 1e-6
                assert not (overlap_xy and overlap_z), (seed, one.package_id, other.package_id)


def test_every_carton_rests_on_the_deck_or_on_the_one_below_it() -> None:
    bay = TruckBay()
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        by_column: dict[int, list[TruckSlot]] = {}
        for slot in slots:
            by_column.setdefault(slot.column, []).append(slot)
        for column in by_column.values():
            column.sort(key=lambda slot: slot.level)
            assert column[0].bottom == pytest.approx(bay.floor_height, abs=1e-9)
            for lower, upper in zip(column, column[1:], strict=False):
                assert upper.bottom == pytest.approx(lower.top, abs=1e-9)
                assert upper.level == lower.level + 1


def _support_ratio(lower: TruckSlot, upper: TruckSlot) -> float:
    l1, r1, n1, f1 = _corners(lower)
    l2, r2, n2, f2 = _corners(upper)
    overlap = max(0.0, min(r1, r2) - max(l1, l2)) * max(0.0, min(f1, f2) - max(n1, n2))
    return overlap / (upper.footprint[0] * upper.footprint[1])


def test_every_carton_lands_squarely_on_the_one_below_it() -> None:
    """A pile that only half carries its next carton is a pile that comes down."""
    bay = TruckBay()
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        by_column: dict[int, list[TruckSlot]] = {}
        for slot in slots:
            by_column.setdefault(slot.column, []).append(slot)
        for column in by_column.values():
            column.sort(key=lambda slot: slot.level)
            for lower, upper in zip(column, column[1:], strict=False):
                # The jitter is applied after the seat is chosen, so it can eat a little
                # into the margin the planner checked.
                assert _support_ratio(lower, upper) > bay.min_support - 0.05, (
                    seed,
                    upper.package_id,
                    lower.package_id,
                )


def test_the_unload_order_never_digs_a_carton_out_from_underneath() -> None:
    """Taking the highest top left in the bay is what makes the order legal."""
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        left = list(slots)
        for slot in unload_order(slots):
            assert slot.top == max(item.top for item in left), (seed, slot.package_id)
            resting_on_it = [
                item
                for item in left
                if item is not slot
                and item.column == slot.column
                and item.level > slot.level
            ]
            assert not resting_on_it, (seed, slot.package_id)
            left.remove(slot)


def test_the_same_seed_loads_the_same_trailer_and_other_seeds_do_not() -> None:
    packages = generate_boxes(EVAL_BOX_COUNT, seed=EVAL_SEED)
    bay = TruckBay()
    assert plan_truck_load(packages, bay, seed=3) == plan_truck_load(packages, bay, seed=3)
    assert plan_truck_load(packages, bay, seed=3) != plan_truck_load(packages, bay, seed=4)


def test_the_load_is_stacked_rather_than_tipped_in() -> None:
    """Columns line up on the deck, and each one is squarely under the next."""
    bay = TruckBay()
    for seed in _SEEDS:
        _packages, slots = _load(seed)
        columns = {slot.column for slot in slots}
        assert len(columns) >= 2, seed
        for slot in slots:
            assert abs(slot.yaw % 90) <= bay.yaw_jitter_deg or abs(
                slot.yaw % 90 - 90
            ) <= bay.yaw_jitter_deg, (seed, slot.yaw)
        by_column: dict[int, list[TruckSlot]] = {}
        for slot in slots:
            by_column.setdefault(slot.column, []).append(slot)
        wander = bay.max_slide + 2 * bay.position_jitter + 1e-9
        for column in by_column.values():
            column.sort(key=lambda slot: slot.level)
            for lower, upper in zip(column, column[1:], strict=False):
                assert abs(upper.center[0] - lower.center[0]) <= wander
                assert abs(upper.center[1] - lower.center[1]) <= wander


def test_the_eval_parcels_fit_the_trailer_the_cell_ships_with() -> None:
    for seed in _SEEDS:
        packages = generate_boxes(EVAL_BOX_COUNT, seed=seed)
        slots = plan_truck_load(packages, TruckBay(), seed=seed)
        assert len(slots) == EVAL_BOX_COUNT


def test_a_load_that_does_not_fit_is_refused_instead_of_stacked_out_of_reach() -> None:
    oversized = tuple(
        Package(f"SLAB-{index}", (0.55, 0.40, 0.30), 6.0) for index in range(12)
    )
    with pytest.raises(TruckLoadTooBig):
        plan_truck_load(oversized, TruckBay(), seed=0)


def test_the_summary_reports_the_load_and_the_order_it_comes_out_in() -> None:
    packages, slots = _load(EVAL_SEED)
    summary = load_summary(slots, TruckBay())

    assert summary["columns"] == len({slot.column for slot in slots})
    assert summary["planned_pick_order"] == [slot.package_id for slot in unload_order(slots)]
    assert len(summary["slots"]) == len(packages)
    assert summary["tallest_column_m"] == pytest.approx(
        max(slot.top for slot in slots) - TruckBay().floor_height
    )
