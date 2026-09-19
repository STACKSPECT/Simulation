"""Measure the centre of mass of a package the robot is holding.

The cell already knows what each package weighs because the scenario says so. A real
infeed does not: mass and balance are exactly what nobody wrote on the box, and a package
whose weight sits off to one side is the one that topples a stack or peels off the cups.
This probe recovers both from the wrist force/torque sensor alone.

The cycle is: tare the empty tool, take the package, then hold it still in several wrist
orientations and solve the static wrench balance of `stable_pallet.com_estimator`. Gravity
in the sensor frame comes from forward kinematics, so nothing here reads simulator truth.

Two details decide whether this works at all:

- **The orientations have to tilt the tool.** `PalletizingSimulator._solve_ik` only aims
  the tool straight down with a yaw, and gravity expressed in the sensor frame is then
  identical in every pose. That leaves the system at rank 2 and the centre of mass is not
  observable. So the probe drives the wrist joints directly instead of going through IK,
  with the same joint-space smoothstep `_move_tool` uses for palletizing.
- **The grasp has to be rigid.** Holding the package with a weld equality biases the
  torque the sensor reports: measured on this cell it costs between 50 mm and 750 mm of
  centre-of-mass error depending on `torquescale`, and it is not a settling artefact --
  it survives full settling and stiffer constraints. Sealing therefore rebuilds the cell
  with the package parented to the tool, which is what a sealed vacuum cup is anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .com_estimator import ComEstimate, estimate_com
from .models import Package
from .simulator import HeldPackage, PalletizingSimulator
from .tare import TareCalibration, calibrate_tare
from .wrench import average_wrench, read_wrench

# Wrist offsets from the resting pose, in radians. The first is the tool hanging straight
# down; the rest tilt it so gravity points somewhere else in the sensor frame. Two
# non-parallel poses already give full rank, the others just average noise away.
MEASUREMENT_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.7, 0.0, 0.0),
    (0.0, 0.9, 0.0),
    (-0.5, 0.6, 0.4),
)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    # Same smoothstep duration as `PalletizingSimulator._move_tool`.
    move_seconds: float = 0.85
    settle_steps: int = 800
    average_steps: int = 120
    # The wrench balance is static: measuring while anything still moves folds inertial
    # terms into the reading and corrupts the moment.
    static_velocity: float = 1e-3  # rad/s and m/s
    lift_clearance: float = 0.35  # m above the pick pose before measuring


@dataclass(frozen=True, slots=True)
class PackageMeasurement:
    """What the probe learned about one package."""

    package_id: str
    mass: float  # kg
    com_tool: np.ndarray  # (3,) m, in the tool frame
    com_local: tuple[float, float, float]  # m from the package centre, in package axes
    com_normalised: tuple[float, float, float]  # fraction of each half-extent
    estimate: ComEstimate

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
                rest_steps=self.config.settle_steps,
            )
        for _ in range(self.config.settle_steps):
            if self._is_static:
                return True
            sim.mujoco.mj_step(sim.model, sim.data)
            sim._capture_frame()
        return self._is_static

    @property
    def _is_static(self) -> bool:
        sim = self.sim
        if np.max(np.abs(sim.data.qvel[sim.arm_dofs])) > self.config.static_velocity:
            return False
        if self._held is None:
            return True
        body = sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, f"package_{self._held.index}")
        return bool(np.max(np.abs(sim.data.cvel[body])) <= self.config.static_velocity)

    def _sample_here(self) -> Any:
        sim = self.sim
        # Inertial wrench from leftover motion folds straight into the moment, so a
        # sample taken while the wrist is still moving is not a CoM measurement.
        for _ in range(self.config.settle_steps):
            if self._is_static:
                break
            sim.mujoco.mj_step(sim.model, sim.data)
            sim._capture_frame()
        else:
            raise RuntimeError("com probe: arm still moving, CoM sample refused")
        batch = []
        for _ in range(self.config.average_steps):
            sim.mujoco.mj_step(sim.model, sim.data)
            sim._capture_frame()
            batch.append(read_wrench(sim.mujoco, sim.model, sim.data))
        return average_wrench(batch)

    def _sweep(self, base: np.ndarray, *, label: str) -> list[Any]:
        samples = []
        for index, offset in enumerate(MEASUREMENT_OFFSETS, start=1):
            self.sim.set_viewer_status(label, f"pose {index}/{len(MEASUREMENT_OFFSETS)}")
            target = base.copy()
            target[3:6] += offset
            if not self._settle(target):
                raise RuntimeError(f"com probe: pose {index} did not come to rest")
            samples.append(self._sample_here())
        return samples

    # -- steps -------------------------------------------------------------------------

    def calibrate(self, base: np.ndarray | None = None) -> TareCalibration:
        """Weigh the empty tool. Describes the tool, not the package, so it is reusable."""
        if self._held is not None:
            raise RuntimeError("calibrate: the tool is holding a package")
        self.sim.set_viewer_status("Taring empty tool", "wrist force/torque")
        home = np.asarray(base if base is not None else self.sim.home, dtype=float)
        return calibrate_tare(self._sweep(home, label="Taring empty tool"))

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

    def measure(self, index: int, tare: TareCalibration, *, lift: bool = True) -> PackageMeasurement:
        """Hold the package in several orientations and solve its centre of mass."""
        if self._held is None or self._held.index != index:
            raise RuntimeError("measure: the package must be grasped first")
        sim = self.sim
        base = np.asarray(sim.data.qpos[sim.arm_qpos]).copy()
        package = sim.scenario.packages[index]
        label = f"Weighing {package.id}"
        if lift:
            # Measuring needs the package clear of everything it could lean on.
            sim.set_viewer_status(label, "lifting clear")
            base[1] -= self.config.lift_clearance
            self._settle(base)
            base = np.asarray(sim.data.qpos[sim.arm_qpos]).copy()

        estimate = estimate_com(self._sweep(base, label=label), tare)
        com_local = self._to_package_frame(estimate.com)
        half = np.asarray(package.size, dtype=float) / 2.0
        normalised = com_local / half
        return PackageMeasurement(
            package_id=package.id,
            mass=estimate.mass,
            com_tool=estimate.com,
            com_local=tuple(float(value) for value in com_local),
            com_normalised=tuple(float(value) for value in normalised),
            estimate=estimate,
        )

    def _to_package_frame(self, com_tool: np.ndarray) -> np.ndarray:
        """Convert from the sensor frame to the package's own axes.

        No perception needed: the cup sealed at a pose the cell chose, so the transform is
        known exactly. The sensor site sits at the tool origin with no rotation, which
        makes the tool frame and the sensor frame the same frame.
        """
        held = self._held
        assert held is not None
        rotation = np.empty(9)
        self.sim.mujoco.mju_quat2Mat(rotation, np.asarray(held.quaternion, dtype=float))
        return rotation.reshape(3, 3).T @ (com_tool - np.asarray(held.position, dtype=float))
