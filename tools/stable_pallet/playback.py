"""Record the cell as it runs so the operator can pause, step back and replay it.

A palletizing run is a one-way loop: plan, pick, place, measure, plan again. That is
fine until something interesting happens in half a second and there is no way to look at
it again. This module keeps a rolling record of the cell -- MuJoCo state, the colours the
run has set, and whether a package is currently welded to the tool -- and can put any of
those frames back on screen.

Scrubbing is deliberately a review, not a rewind of the run itself. The Python side of
the loop (which carton is next, what the planner has decided) cannot be unwound with it,
so stepping back shows you history while the run stays parked exactly where it was, and
resuming continues from there. To make that safe the first scrub takes a full snapshot of
the live cell, which is put back the moment the operator presses play.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .controls import ViewerControls

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .simulator import HeldPackage, PalletizingSimulator

# Everything needed to put a frame back on screen. Rendering only reads positions, so
# the forces and controls that drove the cell there are left out; they are an order of
# magnitude more memory and no pixel depends on them.
FRAME_FIELDS = ("mjSTATE_PHYSICS", "mjSTATE_MOCAP_POS", "mjSTATE_MOCAP_QUAT")
# A live snapshot has to be good enough to carry on stepping from, which is a different
# bar: the run must not notice that the operator looked away.
LIVE_FIELDS = ("mjSTATE_INTEGRATION", "mjSTATE_EQ_ACTIVE")

REVIEW_REFRESH = 1 / 60  # seconds between viewer updates while parked
UNPACED_PLAYBACK_RATE = 8.0  # how fast "max" replays, in multiples of real time


def _signature(mujoco: Any, fields: tuple[str, ...]) -> int:
    signature = 0
    for name in fields:
        signature |= int(getattr(mujoco.mjtState, name))
    return signature


@dataclass(frozen=True, slots=True)
class Frame:
    """One recorded instant of the cell."""

    state: np.ndarray
    appearance: np.ndarray
    held: HeldPackage | None
    status: tuple[str, str]
    sim_time: float


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    """The live cell, complete enough to resume stepping after a detour through history."""

    state: np.ndarray
    held: HeldPackage | None
    status: tuple[str, str]
    eq_data: np.ndarray
    jnt_limited: np.ndarray
    jnt_range: np.ndarray
    geom_rgba: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray


class Playback:
    """Records the cell and serves pause, scrub and replay against that record."""

    def __init__(
        self,
        simulator: PalletizingSimulator,
        controls: ViewerControls | None = None,
        *,
        interval: float = 0.05,
        capacity: int = 20_000,
    ) -> None:
        self.sim = simulator
        self.controls = controls or simulator.controls
        self.interval = interval
        self.capacity = capacity
        self.frames: list[Frame] = []
        self.stride = max(1, round(interval / simulator.model.opt.timestep))
        self._frame_signature = _signature(simulator.mujoco, FRAME_FIELDS)
        self._live_signature = _signature(simulator.mujoco, LIVE_FIELDS)
        self._steps_since_frame = 0
        self._appearance_names = [*simulator._appearance_geom_names(), "beam"]
        self._appearance_cache: tuple[int, list[int]] | None = None
        self._shown: int | None = None
        self._carry = 0.0
        self._last_tick = time.monotonic()

    # -- recording ----------------------------------------------------------------

    def record(self) -> None:
        """Store a frame every `interval` of simulated time.

        Counted in steps rather than `data.time`, because recompiling the cell on a
        grasp hands back a fresh `MjData` whose clock starts again at zero.
        """
        self._steps_since_frame += 1
        stride = self.stride * (6 if self.sim._pacing_suspended else 1)
        if self._steps_since_frame < stride:
            return
        self._steps_since_frame = 0
        sim = self.sim
        state = np.empty(sim.mujoco.mj_stateSize(sim.model, self._frame_signature))
        sim.mujoco.mj_getState(sim.model, sim.data, state, self._frame_signature)
        frame = Frame(
            state=state,
            appearance=sim.model.geom_rgba[self._appearance_ids()].astype(np.float32),
            held=sim.held_package,
            status=sim.viewer_status,
            sim_time=float(sim.data.time),
        )
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            del self.frames[0]
        self.controls.frame_count = len(self.frames)

    # -- pausing and replaying ------------------------------------------------------

    def gate(self) -> None:
        """Block while the operator has the run paused, then hand the cell back intact."""
        controls = self.controls
        if not controls.paused or controls.cancelled:
            return
        live: LiveSnapshot | None = None
        self._begin_review()
        while controls.paused and not controls.cancelled:
            if live is None and controls.review_index is not None:
                live = self.snapshot()
            self._serve()
            time.sleep(REVIEW_REFRESH)
        if live is not None:
            self.restore_snapshot(live)
            self._shown = None

    def hold(self) -> None:
        """Park a finished run on its last frame so it can be replayed before closing."""
        controls = self.controls
        if self.sim.viewer is None or not controls.hold_at_end or controls.cancelled or not self.frames:
            return
        controls.holding = True
        controls.paused = True
        controls.review_index = len(self.frames) - 1
        self._begin_review()
        while controls.holding and not controls.cancelled:
            self._serve()
            time.sleep(REVIEW_REFRESH)
        controls.holding = False
        controls.paused = False

    def _begin_review(self) -> None:
        self._carry = 0.0
        self._last_tick = time.monotonic()

    def _serve(self) -> None:
        """One tick of review: advance any running playback, show the wanted frame."""
        self._advance_playback()
        target = self.controls.review_index
        if target is not None and target != self._shown and self.frames:
            self.restore(target)
            self._shown = target
        self.sim.sync_viewer()

    def _advance_playback(self) -> None:
        controls = self.controls
        now = time.monotonic()
        elapsed, self._last_tick = now - self._last_tick, now
        if controls.playback == "stopped" or not self.frames:
            return
        speed = controls.speed if controls.speed > 0 else UNPACED_PLAYBACK_RATE
        self._carry += speed / self.interval * elapsed
        steps = int(self._carry)
        if not steps:
            return
        self._carry -= steps
        current = controls.review_index if controls.review_index is not None else len(self.frames) - 1
        step = steps if controls.playback == "forward" else -steps
        target = current + step
        if target <= 0:
            target, controls.playback = 0, "stopped"
        elif target >= len(self.frames) - 1:
            target, controls.playback = len(self.frames) - 1, "stopped"
        controls.review_index = target

    # -- putting the cell somewhere else --------------------------------------------

    def restore(self, index: int) -> None:
        """Show the recorded frame at `index`."""
        frame = self.frames[max(0, min(index, len(self.frames) - 1))]
        sim = self.sim
        self._match_held(frame.held)
        sim.mujoco.mj_setState(sim.model, sim.data, frame.state, self._frame_signature)
        sim.model.geom_rgba[self._appearance_ids()] = frame.appearance
        sim.mujoco.mj_forward(sim.model, sim.data)
        sim.set_viewer_status(*frame.status)

    def snapshot(self) -> LiveSnapshot:
        sim = self.sim
        state = np.empty(sim.mujoco.mj_stateSize(sim.model, self._live_signature))
        sim.mujoco.mj_getState(sim.model, sim.data, state, self._live_signature)
        return LiveSnapshot(
            state=state,
            held=sim.held_package,
            status=sim.viewer_status,
            eq_data=sim.model.eq_data.copy(),
            jnt_limited=sim.model.jnt_limited.copy(),
            jnt_range=sim.model.jnt_range.copy(),
            geom_rgba=sim.model.geom_rgba.copy(),
            geom_contype=sim.model.geom_contype.copy(),
            geom_conaffinity=sim.model.geom_conaffinity.copy(),
        )

    def restore_snapshot(self, snapshot: LiveSnapshot) -> None:
        """Undo a review: the run must find the cell exactly as it left it.

        The model carries as much of the run's state as `MjData` does -- the weld poses,
        which pallet joints are locked, which cups are lit -- and recompiling on a grasp
        resets all of it to the XML, so it is restored alongside the state vector.
        """
        sim = self.sim
        self._match_held(snapshot.held)
        sim.model.eq_data[:] = snapshot.eq_data
        sim.model.jnt_limited[:] = snapshot.jnt_limited
        sim.model.jnt_range[:] = snapshot.jnt_range
        sim.model.geom_rgba[:] = snapshot.geom_rgba
        sim.model.geom_contype[:] = snapshot.geom_contype
        sim.model.geom_conaffinity[:] = snapshot.geom_conaffinity
        sim.mujoco.mj_setState(sim.model, sim.data, snapshot.state, self._live_signature)
        sim.mujoco.mj_forward(sim.model, sim.data)
        sim.set_viewer_status(*snapshot.status)

    def _match_held(self, held: HeldPackage | None) -> None:
        """Recompile only when the frame disagrees about what the tool is carrying."""
        current = self.sim.held_package
        if (current is None) == (held is None) and (current is None or current.index == held.index):
            return
        self.sim.rebuild(held)
        self._appearance_cache = None

    def _appearance_ids(self) -> list[int]:
        model = self.sim.model
        if self._appearance_cache is not None and self._appearance_cache[0] == id(model):
            return self._appearance_cache[1]
        ids = [self.sim._geom_id(name) for name in self._appearance_names]
        ids = [geom for geom in ids if geom >= 0]
        self._appearance_cache = (id(model), ids)
        return ids
