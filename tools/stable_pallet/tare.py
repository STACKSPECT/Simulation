"""Tare calibration of the vacuum tool plus sensor.

Before anything can be measured about a payload, whatever the sensor always sees has to
be subtracted: the weight of the tool itself, the moment its own centre of mass produces,
and the constant offset of the transducer. With the tool empty and the model in
`stable_pallet.wrench`:

    F_measured   = -m_t * g_s + b_F
    tau_measured = [g_s]_x c_t + b_tau        with c_t = m_t * r_t

Both blocks are linear. The force block has 4 unknowns (`m_t`, `b_F`) and each pose gives
3 equations, so it needs >= 2 poses with non-parallel gravity. The torque block has 6
(`c_t`, `b_tau`) and with only two poses its null space is the direction `g_2 - g_1`, so
it needs >= 3 poses whose gravity differences are not collinear.

This matters more than any refinement of the estimator: tare error is systematic, it does
not shrink by averaging, and it propagates almost linearly into the estimated centre of
mass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .wrench import WrenchSample, skew

MIN_FORCE_POSES = 2
MIN_TORQUE_POSES = 3
# Above this the pose set is ill-conditioned and the tare cannot be trusted.
MAX_CONDITION = 1.0e3

# Wrist joint offsets from a plumb pose, in radians. The first hangs straight down;
# the rest tilt the tool so gravity points somewhere else in the sensor frame. The
# tare needs the tilts; `--precise-com` reuses them for a full-rank payload solve.
WRIST_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.7, 0.0, 0.0),
    (0.0, 0.9, 0.0),
    (-0.5, 0.6, 0.4),
)


@dataclass(frozen=True, slots=True)
class TareCalibration:
    """What the sensor reads with the tool empty."""

    mass: float  # kg, mass of tool and adapter
    moment: np.ndarray  # (3,) kg*m, first mass moment m*r of that mass
    force_offset: np.ndarray  # (3,) N, constant transducer offset
    torque_offset: np.ndarray  # (3,) Nm
    force_condition: float
    torque_condition: float
    force_residual: float  # N, rms residual
    torque_residual: float  # Nm, rms residual

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("TareCalibration: tool mass must be positive")
        for name in ("moment", "force_offset", "torque_offset"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(3)
            object.__setattr__(self, name, value)

    @property
    def com(self) -> np.ndarray:
        """Centre of mass of the tool in the sensor frame, m."""
        return self.moment / self.mass

    @property
    def well_conditioned(self) -> bool:
        return max(self.force_condition, self.torque_condition) < MAX_CONDITION

    def subtract(self, sample: WrenchSample) -> WrenchSample:
        """Return the wrench attributable to the payload alone."""
        force = sample.force - (-self.mass * sample.gravity + self.force_offset)
        torque = sample.torque - (skew(sample.gravity) @ self.moment + self.torque_offset)
        return WrenchSample(force=force, torque=torque, gravity=sample.gravity)


def calibrate_tare(samples: Sequence[WrenchSample]) -> TareCalibration:
    """Fit tool mass, its moment and the transducer offsets from empty-tool readings."""
    if len(samples) < MIN_TORQUE_POSES:
        raise ValueError(f"calibrate_tare: need >= {MIN_TORQUE_POSES} poses, got {len(samples)}")

    eye = np.eye(3)
    force_matrix = np.vstack([np.hstack([-sample.gravity.reshape(3, 1), eye]) for sample in samples])
    force_target = np.concatenate([sample.force for sample in samples])
    force_solution, *_ = np.linalg.lstsq(force_matrix, force_target, rcond=None)

    torque_matrix = np.vstack([np.hstack([skew(sample.gravity), eye]) for sample in samples])
    torque_target = np.concatenate([sample.torque for sample in samples])
    torque_solution, *_ = np.linalg.lstsq(torque_matrix, torque_target, rcond=None)

    mass = float(force_solution[0])
    if mass <= 0.0:
        raise ValueError("calibrate_tare: non-positive tool mass, check signs or poses")

    return TareCalibration(
        mass=mass,
        moment=torque_solution[:3],
        force_offset=force_solution[1:],
        torque_offset=torque_solution[3:],
        force_condition=float(np.linalg.cond(force_matrix)),
        torque_condition=float(np.linalg.cond(torque_matrix)),
        force_residual=_rms(force_matrix @ force_solution - force_target),
        torque_residual=_rms(torque_matrix @ torque_solution - torque_target),
    )


def _rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))
