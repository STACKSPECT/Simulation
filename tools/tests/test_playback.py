import threading

import mujoco
import numpy as np
import pytest

from stable_pallet.controls import RunCancelled, ViewerControls
from stable_pallet.playback import Playback
from stable_pallet.scenario import load_scenario
from stable_pallet.simulator import HeldPackage, PalletizingSimulator


@pytest.fixture
def cell():
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
    )
    try:
        yield simulator
    finally:
        simulator.close()


def _attach(simulator: PalletizingSimulator, **kwargs) -> Playback:
    playback = Playback(simulator, interval=kwargs.pop("interval", 0.01), **kwargs)
    simulator.playback = playback
    return playback


def _package_position(simulator: PalletizingSimulator, index: int) -> np.ndarray:
    """Where a package is, with the kinematics brought up to date with the state.

    `mj_step` leaves `xpos` one step behind `qpos`, so reading it straight after a step
    and after a restore would compare two different instants.
    """
    mujoco.mj_forward(simulator.model, simulator.data)
    body = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
    return simulator.data.xpos[body].copy()


def _record_now(playback: Playback) -> int:
    """Force a frame for the current instant and return its index."""
    for _ in range(playback.stride):
        playback.record()
    return len(playback.frames) - 1


def test_stepping_the_cell_fills_the_timeline(cell) -> None:
    playback = _attach(cell)
    cell._step(0.2)
    assert len(playback.frames) == pytest.approx(0.2 / 0.01, abs=2)
    assert cell.controls.frame_count == len(playback.frames)


def test_the_recording_is_counted_in_steps_not_in_the_clock(cell) -> None:
    """Recompiling on a grasp hands back a fresh `MjData` whose clock restarts at zero."""
    playback = _attach(cell)
    cell._step(0.1)
    recorded = len(playback.frames)
    cell.rebuild(None)
    assert cell.data.time == 0.0

    cell._step(0.1)

    assert len(playback.frames) > recorded


def test_restoring_a_frame_puts_the_cartons_back_where_they_were(cell) -> None:
    playback = _attach(cell)
    cell._set_package_pose(0, (-0.85, 0.0, 0.95))
    cell._step(0.1)
    early = _record_now(playback)
    early_position = _package_position(cell, 0)

    cell._step(0.5)
    assert not np.allclose(_package_position(cell, 0), early_position, atol=1e-3)

    playback.restore(early)

    assert _package_position(cell, 0) == pytest.approx(early_position, abs=1e-9)


def test_a_frame_carries_the_colours_the_run_had_set(cell) -> None:
    playback = _attach(cell)
    cell._step(0.05)
    hidden = _record_now(playback)
    geom = cell._package_geom(1)
    assert cell.model.geom_rgba[geom][3] == 0.0

    cell._set_package_pose(1, (-0.85, 0.0, 0.95))
    cell._step(0.1)
    assert cell.model.geom_rgba[geom][3] == 1.0

    playback.restore(hidden)

    assert cell.model.geom_rgba[geom][3] == 0.0


def test_reviewing_across_a_grasp_recompiles_the_cell(cell) -> None:
    """A carried package is part of the tool, so the two moments are different models."""
    playback = _attach(cell)
    cell._set_package_pose(2, (-0.85, 0.0, 0.95))
    cell._step(0.1)
    free = _record_now(playback)

    cell.rebuild(HeldPackage(2, (0.0, 0.0, -0.1), (1.0, 0.0, 0.0, 0.0)))
    cell._step(0.1)
    assert cell.held_package is not None

    playback.restore(free)

    assert cell.held_package is None
    assert mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_JOINT, "package_joint_2") >= 0


def test_a_detour_through_history_leaves_the_run_exactly_as_it_was(cell) -> None:
    """Resuming has to continue from the live cell, not from whatever was on screen."""
    playback = _attach(cell)
    cell._set_package_pose(0, (-0.85, 0.0, 0.95))
    cell._step(0.3)
    snapshot = playback.snapshot()
    live_qpos = cell.data.qpos.copy()
    live_position = _package_position(cell, 0)

    playback.restore(0)
    assert not np.allclose(cell.data.qpos, live_qpos, atol=1e-6)

    playback.restore_snapshot(snapshot)

    assert cell.data.qpos == pytest.approx(live_qpos, abs=1e-9)
    assert _package_position(cell, 0) == pytest.approx(live_position, abs=1e-9)


def test_the_live_snapshot_carries_the_model_state_a_recompile_would_reset(cell) -> None:
    playback = _attach(cell)
    cell._set_package_pose(3, (-0.85, 0.0, 0.95))
    cell._step(0.05)
    cell._set_suction(3, True)
    cell._configure_pallet_joints(beam=True)
    snapshot = playback.snapshot()
    weld = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_3")
    relpose = cell.model.eq_data[weld, 3:10].copy()

    cell.rebuild(None)
    assert cell.model.eq_data[weld, 3:10] != pytest.approx(relpose)

    playback.restore_snapshot(snapshot)

    assert cell.model.eq_data[weld, 3:10] == pytest.approx(relpose)
    assert bool(cell.data.eq_active[weld])
    assert not cell.model.jnt_limited[cell._joint_id("pallet_rx")]


def test_the_timeline_forgets_the_oldest_frames_rather_than_growing_without_bound(cell) -> None:
    playback = _attach(cell, capacity=5)
    cell._step(0.3)
    assert len(playback.frames) == 5


def test_a_cancelled_run_stops_at_the_next_frame(cell) -> None:
    cell.controls = ViewerControls()
    cell.controls.cancel()
    with pytest.raises(RunCancelled):
        cell._step(0.02)


def test_pausing_without_scrubbing_does_not_disturb_the_cell(cell) -> None:
    """The gate has to be free: a pause that moved anything would change the result."""
    playback = _attach(cell)
    cell._step(0.05)
    before = cell.data.qpos.copy()

    cell.controls.paused = True
    threading.Timer(0.15, setattr, (cell.controls, "paused", False)).start()
    playback.gate()

    assert cell.data.qpos == pytest.approx(before, abs=1e-12)


def test_a_pause_that_scrubs_hands_the_live_cell_back_on_resume(cell) -> None:
    playback = _attach(cell)
    cell._set_package_pose(0, (-0.85, 0.0, 0.95))
    cell._step(0.3)
    live_qpos = cell.data.qpos.copy()

    cell.controls.scrub_to(0)
    threading.Timer(0.25, cell.controls.resume).start()
    playback.gate()

    assert cell.data.qpos == pytest.approx(live_qpos, abs=1e-9)
