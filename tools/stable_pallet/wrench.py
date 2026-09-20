"""Wrist force/torque readings and the static gravity model behind them.

The sensor reports the wrench crossing the wrist-to-tool joint, expressed in the site
frame. At rest that wrench holds up everything distal to it (tool plus payload), so for a
mass `m` whose centre of mass sits at `r` from the sensor origin:

    F   = -m * g_s                 (the flange pushes up to carry the weight)
    tau = [g_s]_x (m * r) = r x F  (and cancels the gravitational moment)

where `g_s` is gravity expressed in the sensor frame. Both forms are the same equation:
the first is linear in the first mass moment `m * r` and drives the tare calibration, the
second is linear in `r` and drives the payload centre-of-mass estimate.

`g_s` comes from forward kinematics (the site orientation) and the model gravity vector,
so it is observable information rather than simulator ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FORCE_SENSOR = "ft_force"
TORQUE_SENSOR = "ft_torque"
SENSOR_SITE = "ft_site"

# Spread allowed between the gravity vectors of one batch, m/s2. Two measurement poses
# differ by several m/s2, so this separates "same pose, arm still settling" from "I mixed
# two poses together", which is the mistake worth catching.
#
# A quick in-situ reading still creeps a few milliradians; a threshold of ~0.06° fired
# every time. ~0.3° is harmless for a single pose and still catches a mixed pair.
SAME_POSE_TOLERANCE = 9.81 * math.sin(math.radians(0.3))


def skew(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that `skew(a) @ b == np.cross(a, b)`."""
    x, y, z = (float(component) for component in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True, slots=True)
class WrenchSample:
    """One force/torque reading together with the orientation it was taken in."""

    force: np.ndarray  # (3,) N, in the sensor frame
    torque: np.ndarray  # (3,) Nm, in the sensor frame
    gravity: np.ndarray  # (3,) m/s2, gravity in the sensor frame

    def __post_init__(self) -> None:
        for name in ("force", "torque", "gravity"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"WrenchSample: {name} must be finite")
            object.__setattr__(self, name, value)
        if float(np.linalg.norm(self.gravity)) <= 0.0:
            raise ValueError("WrenchSample: gravity cannot be zero")

    @property
    def gravity_norm(self) -> float:
        return float(np.linalg.norm(self.gravity))


def read_wrench(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    site: str = SENSOR_SITE,
    force_sensor: str = FORCE_SENSOR,
    torque_sensor: str = TORQUE_SENSOR,
) -> WrenchSample:
    """Read the wrist force/torque sensor in the simulator's current state."""
    force_address = model.sensor(force_sensor).adr[0]
    torque_address = model.sensor(torque_sensor).adr[0]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    rotation = data.site_xmat[site_id].reshape(3, 3)
    return WrenchSample(
        force=data.sensordata[force_address : force_address + 3].copy(),
        torque=data.sensordata[torque_address : torque_address + 3].copy(),
        gravity=rotation.T @ np.asarray(model.opt.gravity, dtype=float),
    )


def average_wrench(samples: Sequence[WrenchSample], tolerance: float = SAME_POSE_TOLERANCE) -> WrenchSample:
    """Average readings taken in one pose.

    White sensor noise falls with the square root of the sample count, so averaging a few
    milliseconds already pushes it below the systematic terms. What averaging cannot
    remove is the tare, which is what `stable_pallet.tare` is for.
    """
    if not samples:
        raise ValueError("average_wrench: no samples given")
    gravities = np.stack([sample.gravity for sample in samples])
    spread = float(np.max(np.linalg.norm(gravities - gravities.mean(axis=0), axis=1)))
    if spread > tolerance:
        raise ValueError("average_wrench: samples do not come from the same pose")
    return WrenchSample(
        force=np.mean([sample.force for sample in samples], axis=0),
        torque=np.mean([sample.torque for sample in samples], axis=0),
        gravity=gravities.mean(axis=0),
    )
