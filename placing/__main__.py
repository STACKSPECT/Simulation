"""Self-check for the placement heuristic: `python -m placing`.

Asserts only, no test framework and no simulator, so it runs anywhere the module
runs. If this passes, the module is doing what its callers assume.
"""

from __future__ import annotations

import sys

import numpy as np

from placing import (
    METRIC_NAMES,
    BoxOrder,
    FeasibilityConfig,
    PalletState,
    PlacementPlanner,
    ScoringConfig,
    choose,
    empty_height_map,
    plan,
    reject_summary,
    resample_height_map,
    yaw_set,
)

L, W = 0.60, 0.40


def _flat(res: float, height: float = 0.0):
    """Fully observed pallet at a uniform height."""
    hmap = empty_height_map(L, W, res)
    hmap.observed[:] = True
    hmap.heights[:] = height
    return hmap


def _planner(weights=None, **feas):
    feas.setdefault("min_support_ratio", 0.0)
    return PlacementPlanner(
        feasibility=FeasibilityConfig(**feas),
        scoring=ScoringConfig(weights=weights or {"support_ratio": 1.0}),
        resolution=None,
    )


def check_boundary() -> None:
    """The module must not drag a simulator, a plotter or a config loader in.

    This is the property that lets the heuristic be tuned without MuJoCo, and it
    is the kind of thing that regresses through one convenience import.
    """
    leaked = sorted(
        m for m in sys.modules if m.startswith(("mujoco", "matplotlib", "yaml", "scipy"))
    )
    assert not leaked, f"placing pulled in {leaked}"


def check_empty_pallet() -> None:
    """Everything fits, everything is fully supported, ties break lower-left."""
    result = plan(_flat(0.01), BoxOrder("b", (0.20, 0.10, 0.10)), planner=_planner())
    best, batch = result.best, result.batch
    assert best is not None
    assert abs(best.support_ratio - 1.0) < 1e-9
    assert abs(best.z_base) < 1e-12
    assert best.yaw == 0.0
    valid = batch.valid
    assert abs(best.y - batch.y[valid].min()) < 1e-9
    lowest = valid & np.isclose(batch.y, batch.y[valid].min())
    assert abs(best.x - batch.x[lowest].min()) < 1e-9


def check_clearance_band() -> None:
    """A neighbour inside the clearance band is a collision; just outside it is not."""
    res = 0.005
    hmap = _flat(res)
    hmap.heights[20:60, 40:60] = 0.40  # a tall block over x cells 40..59
    box = BoxOrder("b", (0.05, 0.05, 0.10))
    cfg = FeasibilityConfig(clearance=res, yaws=(0.0,), min_support_ratio=0.0)
    batch = plan(hmap, box, planner=PlacementPlanner(cfg, resolution=None)).batch
    row = np.abs(batch.y - 0.20) < 0.02
    touching = row & (batch.i0 == 60)
    spaced = row & (batch.i0 == 61)
    assert touching.any() and spaced.any()
    assert np.all(batch.collision[touching]), "touching the block must collide"
    assert not np.any(batch.valid[touching])
    assert not np.any(batch.collision[spaced]), "one clear cell must be enough"
    assert np.all(batch.valid[spaced])


def check_notch_beats_plateau() -> None:
    """A box that fills a hole beats one that starts a new layer on the flat."""
    hmap = _flat(0.01, height=0.24)
    xs, ys = hmap.cell_centers()
    notch = (np.abs(xs - 0.30) < 0.14) & (np.abs(ys - 0.20) < 0.14)
    hmap.heights[notch] = 0.12
    planner = _planner({"void_fill": 1.0, "lowness": 1.0, "support_ratio": 0.2})
    best = choose(hmap, BoxOrder("b", (0.20, 0.20, 0.10)), planner=planner)
    assert best is not None
    assert abs(best.x - 0.30) < 0.14 and abs(best.y - 0.20) < 0.14
    assert abs(best.z_base - 0.12) < 0.011


def check_scores_are_bounded() -> None:
    """Every metric and the score itself live in [0, 1]. grunt's contract needs it."""
    hmap = _flat(0.01)
    hmap.heights[10:30, 15:40] = 0.20
    planner = _planner({name: 1.0 for name in METRIC_NAMES}, yaws=(0.0, 90.0))
    result = plan(hmap, BoxOrder("b", (0.20, 0.15, 0.12)), planner=planner)
    assert result.n_valid > 0
    for name in METRIC_NAMES:
        values = result.metrics[name][result.batch.valid]
        assert np.all(values >= -1e-9) and np.all(values <= 1.0 + 1e-9), name
    scores = result.scores[result.batch.valid]
    assert np.all(scores >= -1e-9) and np.all(scores <= 1.0 + 1e-9)
    assert 0.0 <= result.best.score <= 1.0


def check_non_finite_cells_are_unobserved() -> None:
    """One NaN cell used to turn the whole ranking into NaN, silently.

    The score stayed NaN, nothing was rejected, and the tie-break still handed
    back a pose. Real depth cameras produce NaN, so this has to hold.
    """
    hmap = _flat(0.01)
    hmap.heights[5, 5] = np.nan
    hmap.heights[7, 9] = np.inf
    result = plan(hmap, BoxOrder("b", (0.20, 0.15, 0.12)), planner=_planner())
    assert result.height_map is not None
    assert not result.height_map.observed[5, 5]
    assert not result.height_map.observed[7, 9]
    assert result.best is not None
    assert np.isfinite(result.best.score)
    assert all(np.isfinite(v) for v in result.best.metrics.values())
    assert np.all(np.isfinite(result.scores[result.batch.valid]))
    assert hmap.observed[5, 5], "the caller's map must not be mutated"


def check_offset_com_gets_the_flips() -> None:
    """Turning an off-centre box round is a different placement, so enumerate it."""
    cfg = FeasibilityConfig(yaws=(0.0, 90.0))
    centred = BoxOrder("b", (0.20, 0.15, 0.12))
    offset = BoxOrder("b", (0.20, 0.15, 0.12), com_local=(0.06, -0.03, 0.0))
    assert yaw_set(centred, cfg) == (0.0, 90.0)
    assert set(yaw_set(offset, cfg)) == {0.0, 90.0, 180.0, 270.0}

    hmap = _flat(0.01)
    batch = plan(hmap, offset, planner=_planner()).batch
    assert set(np.unique(batch.yaw)) == {0.0, 90.0, 180.0, 270.0}
    # Same cell, opposite yaw: the CoM lands somewhere else, so it is a real pose.
    corner = (batch.i0 == 0) & (batch.j0 == 0)
    assert np.any(corner & (batch.yaw == 0.0)) and np.any(corner & (batch.yaw == 180.0))


def _two_neighbours(res: float = 0.01, gap: float = 0.06, height: float = 0.20):
    """Two blocks of equal height with a gap between them, along x."""
    hmap = _flat(res)
    xs, ys = hmap.cell_centers()
    left = (xs >= 0.05) & (xs < 0.20)
    right = (xs >= 0.20 + gap) & (xs < 0.35 + gap)
    band = (ys >= 0.10) & (ys < 0.30)
    hmap.heights[left & band] = height
    hmap.heights[right & band] = height
    return hmap


def check_bridge_is_allowed_and_rewarded() -> None:
    """A box laid across two neighbours is stable, so it must be legal and scored.

    It used to be rejected outright: the filter asked whether the cell under the
    CoG was solid, and for a bridge it is thin air. A plank on two trestles is
    the counter-example.
    """
    hmap = _two_neighbours()
    box = BoxOrder("bridge", (0.36, 0.20, 0.10))
    result = plan(hmap, box, planner=_planner(clearance=0.0))
    batch = result.batch
    span = np.flatnonzero(
        batch.valid & (batch.yaw == 0.0) & (np.abs(batch.z_base - 0.20) < 1e-6)
    )
    assert span.size, "no pose rests on both neighbours"
    assert np.any(result.metrics["bridge"][span] > 0.5), "spanning is not detected"
    # The planner should actually take it, not merely tolerate it.
    best = choose(hmap, box, planner=_planner(clearance=0.0))
    assert best is not None and abs(best.z_base - 0.20) < 1e-6
    assert best.metrics["bridge"] > 0.5
    assert best.metrics["com_margin"] > 0.0, "a centred bridge has a real margin"


def check_cantilever_is_still_rejected() -> None:
    """Allowing bridges must not allow a box hanging off one neighbour."""
    hmap = _flat(0.01)
    xs, ys = hmap.cell_centers()
    hmap.heights[(xs >= 0.05) & (xs < 0.20) & (ys >= 0.10) & (ys < 0.30)] = 0.20
    box = BoxOrder("over", (0.36, 0.20, 0.10))
    batch = plan(hmap, box, planner=_planner(clearance=0.0)).batch
    on_top = (np.abs(batch.z_base - 0.20) < 1e-6) & (batch.yaw == 0.0)
    # Every accepted pose keeps its CoG over the block it rests on (x < 0.20).
    accepted = batch.valid & on_top
    assert accepted.any()
    assert np.all(batch.x[accepted] < 0.20 + 1e-9), "the CoG drifted off the support"


def check_first_box_takes_the_corner() -> None:
    """On an empty pallet the first box goes flush into a corner, not near one.

    Being one gap-width off the rim leaves a strip nothing can fill. The rim is
    also not a seam, and counting it as one used to push boxes away from it.
    """
    box = BoxOrder("a", (0.20, 0.15, 0.12))
    best = choose(_flat(0.01), box, planner=_planner({name: 1.0 for name in METRIC_NAMES}))
    assert best is not None
    half_l, half_w = 0.10, 0.075
    assert abs(best.x - half_l) < 1e-9, f"x={best.x}, wanted flush at {half_l}"
    assert abs(best.y - half_w) < 1e-9, f"y={best.y}, wanted flush at {half_w}"
    assert best.metrics["gap_waste"] == 1.0


def check_gap_waste_grades_the_leftover() -> None:
    """Flush is perfect, a sliver is waste, and plenty of room is not waste."""
    hmap = _flat(0.01)
    box = BoxOrder("a", (0.20, 0.15, 0.12))
    result = plan(hmap, box, planner=_planner())
    batch, gap = result.batch, result.metrics["gap_waste"]
    bottom_row = batch.valid & (batch.yaw == 0.0) & (np.abs(batch.y - 0.075) < 1e-9)

    def at(x):
        sel = np.flatnonzero(bottom_row & (np.abs(batch.x - x) < 1e-9))
        assert sel.size, f"no candidate at x={x}"
        return float(gap[sel[0]])

    assert at(0.10) == 1.0, "flush against the rim wastes nothing"
    assert at(0.15) < at(0.12) < at(0.10), "a wider sliver must score worse"

    # A small box in the middle leaves usable room all round: not waste.
    small = plan(hmap, BoxOrder("s", (0.10, 0.10, 0.10)), planner=_planner())
    mid = np.flatnonzero(
        small.batch.valid
        & (np.abs(small.batch.x - 0.30) < 1e-9)
        & (np.abs(small.batch.y - 0.20) < 1e-9)
    )
    assert mid.size and small.metrics["gap_waste"][mid[0]] == 1.0


def check_determinism() -> None:
    """Same inputs, same pick. A planner that drifts cannot be debugged from a log."""
    hmap = _flat(0.01)
    hmap.heights[8:26, 12:38] = 0.18
    box = BoxOrder("b", (0.20, 0.10, 0.10))
    planner = _planner({"support_ratio": 1.0, "lowness": 0.5, "void_fill": 0.8})
    first, second = plan(hmap, box, planner=planner), plan(hmap, box, planner=planner)
    assert first.best is not None and second.best is not None
    assert (first.best.x, first.best.y, first.best.yaw) == (
        second.best.x, second.best.y, second.best.yaw
    )
    np.testing.assert_allclose(first.scores, second.scores)


def check_no_fit_is_a_result_not_a_crash() -> None:
    """Nothing fits: `best` is None and the summary says which constraint said no."""
    tall = BoxOrder("b", (0.20, 0.15, 0.50))
    result = plan(_flat(0.01), tall, planner=_planner(max_stack_height=0.30))
    assert result.best is None
    assert choose(_flat(0.01), tall, planner=_planner(max_stack_height=0.30)) is None
    summary = reject_summary(result.batch)
    assert summary, "an empty result must still say why"
    assert summary.get("max_stack_height", 0) > 0
    assert sum(summary.values()) == len(result.batch)

    # A box bigger than the pallet yields no candidates at all, also not a crash.
    empty = plan(_flat(0.01), BoxOrder("b", (2.0, 0.2, 0.1)), planner=_planner())
    assert empty.best is None and len(empty.batch) == 0
    assert reject_summary(empty.batch) == {}


def check_resample_is_a_max_pool() -> None:
    """Coarsening keeps the tallest point, and keeps `observed` honest."""
    fine = empty_height_map(L, W, 0.005)
    fine.observed[:] = True
    fine.heights[:] = 0.10
    fine.heights[3, 5] = 0.42          # one spike inside destination cell (1, 2)
    fine.observed[20:40, 20:40] = False
    coarse = resample_height_map(fine, 0.01)
    assert coarse.heights.shape == (40, 60)
    assert abs(float(coarse.heights[1, 2]) - 0.42) < 1e-6, "a max pool keeps the spike"
    assert abs(float(coarse.heights[0, 0]) - 0.10) < 1e-6
    assert not coarse.observed[15, 15], "a fully unobserved block stays unobserved"
    assert float(coarse.heights[15, 15]) == 0.0
    assert coarse.observed[5, 5]
    # Identical resolution is a no-op, not a rebuild.
    assert resample_height_map(fine, 0.005) is fine


def check_pallet_com_pulls_to_the_centre() -> None:
    """With only the stack-CoM term on, the first box lands near the pallet centre."""
    half = 0.5 * float(np.hypot(L, W))
    state = PalletState(
        mass=0.0, com_xy=(0.5 * L, 0.5 * W),
        pallet_center_xy=(0.5 * L, 0.5 * W), pallet_half_diag=half,
    )
    best = plan(
        _flat(0.01), BoxOrder("b", (0.20, 0.10, 0.10), mass=2.0),
        pallet_state=state, planner=_planner({"pallet_com": 1.0}),
    ).best
    assert best is not None
    assert abs(best.x - 0.5 * L) < 0.12 and abs(best.y - 0.5 * W) < 0.12


def demo() -> None:
    checks = [
        check_boundary,
        check_empty_pallet,
        check_clearance_band,
        check_notch_beats_plateau,
        check_scores_are_bounded,
        check_non_finite_cells_are_unobserved,
        check_offset_com_gets_the_flips,
        check_bridge_is_allowed_and_rewarded,
        check_cantilever_is_still_rejected,
        check_first_box_takes_the_corner,
        check_gap_waste_grades_the_leftover,
        check_determinism,
        check_no_fit_is_a_result_not_a_crash,
        check_resample_is_a_max_pool,
        check_pallet_com_pulls_to_the_centre,
    ]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    demo()
