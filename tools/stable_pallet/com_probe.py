"""Measure the centre of mass of a package the robot is holding.

The cell already knows what each package weighs because the scenario says so. A real
infeed does not: mass and balance are exactly what nobody wrote on the box, and a package
whose weight sits off to one side is the one that topples a stack or peels off the cups.
This probe recovers both from the wrist force/torque sensor alone.

Default cycle: tare the empty tool once, take the package, hold it still with the tool
plumb, and solve the rank-2 wrench balance of `stable_pallet.com_estimator` against the
geometric centre of the carton. The two horizontal components come out exactly; the
vertical one is the prior, which no placement decision in this cell reads.

`--precise-com` still drives the wrist through several tilted orientations so all three
directions resolve. Gravity in the sensor frame comes from forward kinematics, so nothing
here reads simulator truth.

Two details decide whether this works at all:

- **The grasp has to be rigid.** Holding the package with a weld equality biases the
  torque the sensor reports: measured on this cell it costs between 50 mm and 750 mm of
  centre-of-mass error depending on `torquescale`, and it is not a settling artefact --
  it survives full settling and stiffer constraints. Sealing therefore rebuilds the cell
  with the package parented to the tool, which is what a sealed vacuum cup is anyway.
- **The reading pose has to stay plumb.** A joint-space "safety lift" of 0.35 rad on the
  shoulder tilts the tool, the blind axis leaves the carton vertical, and the prior
  leaks into the plane -- the error is exactly `half-height * sin(lean)`. That lift is
  only for the tilted sweep, which is what it existed for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .com_estimator import HIDDEN_LIMIT_M, TILTED, ComEstimate, estimate_com, hidden_planar_error
from .models import Package
from .simulator import HeldPackage, PalletizingSimulator
from .tare import WRIST_OFFSETS, TareCalibration, calibrate_tare
from .wrench import average_wrench, read_wrench

MEASUREMENT_OFFSETS = WRIST_OFFSETS


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    # Same smoothstep duration as `PalletizingSimulator._move_tool`.
    move_seconds: float = 0.85
    settle_steps: int = 800
    # Consecutive steps below `static_velocity`. One frame at the bottom of a wobble is
    # not stillness: that is how a reading of 1371 N showed up on a 8 kg carton.
    static_steps: int = 25
    average_steps: int = 30
    tare_average_steps: int = 120
    # The wrench balance is static: measuring while anything still moves folds inertial
    # terms into the reading and corrupts the moment.
    static_velocity: float = 0.008  # rad/s and m/s, same as the arm's rest_speed
    # Radians on shoulder_lift, only when the wrist is about to tilt.
    lift_clearance: float = 0.35
    precise: bool = False


@dataclass(frozen=True, slots=True)
class PackageMeasurement:
    """What the probe learned about one package."""

    package_id: str
    mass: float  # kg
    com_tool: np.ndarray  # (3,) m, in the tool frame
    com_local: tuple[float, float, float]  # m from the package centre, in package axes
    com_normalised: tuple[float, float, float]  # fraction of each half-extent
    estimate: ComEstimate
    lean: float = 0.0
    hidden_m: float = 0.0
    duration_s: float = 0.0
    method: str = "in_situ"

    @property
    def trustworthy(self) -> bool:
        """Whether the fit is sound and the result lands inside the package."""
        return self.estimate.trustworthy and max(abs(value) for value in self.com_normalised) <= 1.0

    def as_package(self, package: Package) -> Package:
        """The same package with measured mass and centre of mass, ready for the planner."""
        return Package(
            id=package.id,
            size=package.size,
            mass=self.mass,
            com=self.com_local,
            friction=package.friction,
        )


class ComProbe:
    """Drives a `PalletizingSimulator` through a centre-of-mass measurement."""

    def __init__(self, simulator: PalletizingSimulator, config: ProbeConfig | None = None) -> None:
        self.sim = simulator
        self.config = config or ProbeConfig()
        self._held: HeldPackage | None = None

    # -- motion ------------------------------------------------------------------------

    def _wait_static(self) -> bool:
        """True when the wrist (and the carried carton) stayed still for N steps."""
        sim = self.sim
        still = 0
        for _ in range(self.config.settle_steps):
            if self._is_static:
                still += 1
                if still >= self.config.static_steps:
                    return True
            else:
                still = 0
            sim.mujoco.mj_step(sim.model, sim.data)
            sim._capture_frame()
        return still >= self.config.static_steps

    def _settle(self, target: np.ndarray) -> bool:
        sim = self.sim
        if sim.controls.fast_forward:
            # Jump onto the measurement pose instead of driving to it. The reading is
            # still taken from a settled, static arm: the loop below does not return
            # until the wrist has stopped moving, it just gets there sooner.
            sim._snap_to(target)
        else:
            # Arriving is not enough here. A wrist that has stopped by the standard the
            # transit moves use is still creeping into its final droop, and the balance
            # is solved against the pose it comes to rest on, so the probe waits it out.
            sim._move_joints(
                target,
                seconds=self.config.move_seconds,
                rest_speed=self.config.static_velocity,
                rest_steps=self.config.static_steps,
            )
        return self._wait_static()

    @property
    def _is_static(self) -> bool:
        sim = self.sim
        if np.max(np.abs(sim.data.qvel[sim.arm_dofs])) > self.config.static_velocity:
            return False
        if self._held is None:
            return True
        body = sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, f"package_{self._held.index}")
        return bool(np.max(np.abs(sim.data.cvel[body])) <= self.config.static_velocity)

    def _sample_here(self, *, steps: int | None = None) -> Any:
        sim = self.sim
        # Inertial wrench from leftover motion folds straight into the moment, so a
        # sample taken while the wrist is still moving is not a CoM measurement.
        if not self._wait_static():
            raise RuntimeError("com probe: arm still moving, CoM sample refused")
        batch = []
        n_steps = self.config.average_steps if steps is None else steps
        for _ in range(n_steps):
            sim.mujoco.mj_step(sim.model, sim.data)
            sim._capture_frame()
            batch.append(read_wrench(sim.mujoco, sim.model, sim.data))
        return average_wrench(batch)

    def _sweep(self, base: np.ndarray, *, label: str, average_steps: int | None = None) -> list[Any]:
        samples = []
        for index, offset in enumerate(MEASUREMENT_OFFSETS, start=1):
            self.sim.set_viewer_status(label, f"pose {index}/{len(MEASUREMENT_OFFSETS)}")
            target = base.copy()
            target[3:6] += offset
            if not self._settle(target):
                raise RuntimeError(f"com probe: pose {index} did not come to rest")
            samples.append(self._sample_here(steps=average_steps))
        return samples

    def _package_rotation(self) -> np.ndarray:
        held = self._held
        assert held is not None
        rotation = np.empty(9)
        self.sim.mujoco.mju_quat2Mat(rotation, np.asarray(held.quaternion, dtype=float))
        return rotation.reshape(3, 3)

    # -- steps -------------------------------------------------------------------------

    def calibrate(self, base: np.ndarray | None = None) -> TareCalibration:
        """Weigh the empty tool. Describes the tool, not the package, so it is reusable."""
        if self._held is not None:
            raise RuntimeError("calibrate: the tool is holding a package")
        self.sim.set_viewer_status("Taring empty tool", "wrist force/torque")
        home = np.asarray(base if base is not None else self.sim.home, dtype=float)
        return calibrate_tare(self._sweep(home, label="Taring empty tool", average_steps=self.config.tare_average_steps))

    def grasp(self, index: int) -> HeldPackage:
        """Seal onto a package and rebuild the cell with it carried rigidly.

        Everything after this call uses the new model, so `simulator.data` is a different
        object: anything holding the old one has to ask for it again.
        """
        sim = self.sim
        mujoco = sim.mujoco
        body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
        tool_rotation = sim.data.xmat[sim.tool_body_id].reshape(3, 3)
        position = tool_rotation.T @ (sim.data.xpos[body] - sim.data.xpos[sim.tool_body_id])
        inverse_tool = np.empty(4)
        relative = np.empty(4)
        mujoco.mju_negQuat(inverse_tool, sim.data.xquat[sim.tool_body_id])
        mujoco.mju_mulQuat(relative, inverse_tool, sim.data.xquat[body])

        held = HeldPackage(index, tuple(float(value) for value in position), tuple(float(value) for value in relative))
        sim.rebuild(held)
        self._held = held
        return held

    def release(self) -> None:
        """Cut the vacuum: the package becomes a free body again where it now stands."""
        if self._held is None:
            return
        self.sim.rebuild(None)
        self._held = None

    def measure(self, index: int, tare: TareCalibration, *, lift: bool | None = None) -> PackageMeasurement:
        """Read the package. One plumb pose, or a tilted sweep if `precise` is on."""
        if self._held is None or self._held.index != index:
            raise RuntimeError("measure: the package must be grasped first")
        sim = self.sim
        package = sim.scenario.packages[index]
        label = f"Weighing {package.id}"
        precise = self.config.precise
        if lift is None:
            lift = precise
        started = float(sim.data.time)
        base = np.asarray(sim.data.qpos[sim.arm_qpos]).copy()
        if lift:
            # Measuring with a tilted wrist needs the carton clear of everything it
            # could lean on. A joint-space lift is what used to run on every package;
            # it tilts the tool, so it stays off the in-situ path.
            sim.set_viewer_status(label, "lifting clear")
            base[1] -= self.config.lift_clearance
            self._settle(base)
            base = np.asarray(sim.data.qpos[sim.arm_qpos]).copy()

        prior = np.asarray(self._held.position, dtype=float)
        if precise:
            sim.set_viewer_status(label, "tilted sweep")
            samples = self._sweep(base, label=label)
            method = "sweep"
        else:
            sim.set_viewer_status(label, "in situ")
            samples = [self._sample_here()]
            method = "in_situ"

        estimate = estimate_com(samples, tare, prior=prior)
        rotation = self._package_rotation()
        half = np.asarray(package.size, dtype=float) / 2.0
        lean, hidden = 0.0, 0.0
        if estimate.rank < 3:
            lean, hidden = hidden_planar_error(estimate.blind_axis, rotation, half)
            if hidden > HIDDEN_LIMIT_M and not precise:
                sim.set_viewer_status(label, "blind axis tilted, sweeping")
                lifted = np.asarray(sim.data.qpos[sim.arm_qpos]).copy()
                lifted[1] -= self.config.lift_clearance
                self._settle(lifted)
                estimate = estimate_com(
                    self._sweep(np.asarray(sim.data.qpos[sim.arm_qpos]).copy(), label=label),
                    tare,
                    prior=prior,
                )
                method = "sweep"
                if estimate.rank < 3:
                    lean, hidden = hidden_planar_error(estimate.blind_axis, rotation, half)
                else:
                    lean, hidden = 0.0, 0.0
            if hidden > HIDDEN_LIMIT_M and estimate.reason == "ok":
                estimate = replace(estimate, reason=TILTED)

        com_local = rotation.T @ (estimate.com - prior)
        normalised = com_local / half
        return PackageMeasurement(
            package_id=package.id,
            mass=estimate.mass,
            com_tool=estimate.com,
            com_local=tuple(float(value) for value in com_local),
            com_normalised=tuple(float(value) for value in normalised),
            estimate=estimate,
            lean=lean,
            hidden_m=hidden,
            duration_s=float(sim.data.time) - started,
            method=method,
        )

    def _to_package_frame(self, com_tool: np.ndarray) -> np.ndarray:
        """Convert from the sensor frame to the package's own axes.

        No perception needed: the cup sealed at a pose the cell chose, so the transform is
        known exactly. The sensor site sits at the tool origin with no rotation, which
        makes the tool frame and the sensor frame the same frame.
        """
        held = self._held
        assert held is not None
        prior = np.asarray(held.position, dtype=float)
        return self._package_rotation().T @ (com_tool - prior)
