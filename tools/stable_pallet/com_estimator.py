"""Payload centre-of-mass estimation from a static wrench balance.

With the tare subtracted, what remains at the sensor is the wrench holding the package
alone:

    F_p   = -m_p * g_s
    tau_p = r x F_p = -[F_p]_x r

Each pose contributes three equations but only rank 2: the component of `r` parallel to
gravity produces no moment and is unobservable. That is why one pose is never enough, no
matter how well it is measured. Two poses with non-parallel gravity already give rank 3;
further poses only average noise.

The error scales with the inverse of the mass, because the useful moment grows with the
weight while sensor noise does not. For packages of tens of kilograms the dominant term
stops being noise and becomes the tare residual, which is systematic. Nothing learned is
required: the system is linear and solved in closed form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .tare import TareCalibration
from .wrench import WrenchSample, skew

MIN_POSES = 2
# Above this the poses are nearly parallel and `r` is not observable.
MAX_CONDITION = 1.0e3
# Below this mass the signal cannot be told apart from the tare residual.
MIN_PAYLOAD_MASS = 0.05  # kg
# Relative drift of |F| between poses that already betrays the package having moved.
SLIP_TOLERANCE = 0.02
# Noise sigmas tolerated before calling the |F| spread a slip. Without this margin the
# noise alone exceeds any relative threshold on light payloads and the detector always
# fires.
SLIP_NOISE_MARGIN = 4.0

OK = "ok"
NO_PAYLOAD = "no_payload"
PARALLEL_POSES = "parallel_poses"
SLIPPED = "slipped"


@dataclass(frozen=True, slots=True)
class ComEstimate:
    """Estimated centre of mass and how much it can be trusted."""

    com: np.ndarray  # (3,) m, in the sensor frame
    mass: float  # kg
    condition: float  # conditioning of the pose set
    residual: float  # Nm, rms residual of the fit
    slip: float  # relative drift of |F| between poses
    reason: str  # OK, or why the estimate is not trustworthy

    def __post_init__(self) -> None:
        object.__setattr__(self, "com", np.asarray(self.com, dtype=float).reshape(3))

    @property
    def trustworthy(self) -> bool:
        return self.reason == OK


def estimate_com(
    samples: Sequence[WrenchSample],
    tare: TareCalibration,
    *,
    slip_tolerance: float = SLIP_TOLERANCE,
    force_noise: float = 0.0,
) -> ComEstimate:
    """Solve the package centre of mass from several static poses."""
    if len(samples) < MIN_POSES:
        raise ValueError(f"estimate_com: need >= {MIN_POSES} poses, got {len(samples)}")

    payload = [tare.subtract(sample) for sample in samples]
    magnitudes = np.array([float(np.linalg.norm(item.force)) for item in payload])
    gravity_norm = float(np.mean([sample.gravity_norm for sample in samples]))
    mass = float(np.mean(magnitudes) / gravity_norm)

    # If the package slips between poses the wrench stops describing one rigid body held
    # at a fixed point, and the fit mixes two different geometries.
    slip = float((magnitudes.max() - magnitudes.min()) / max(magnitudes.mean(), 1e-9))

    matrix = np.vstack([-skew(item.force) for item in payload])
    target = np.concatenate([item.torque for item in payload])
    com, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    return ComEstimate(
        com=com,
        mass=mass,
        condition=float(np.linalg.cond(matrix)),
        residual=float(np.sqrt(np.mean(np.square(matrix @ com - target)))),
        slip=slip,
        reason=_verdict(mass, float(np.linalg.cond(matrix)), magnitudes, slip_tolerance, force_noise),
    )


def _verdict(mass: float, condition: float, magnitudes: np.ndarray, slip_tolerance: float, force_noise: float) -> str:
    if mass < MIN_PAYLOAD_MASS:
        return NO_PAYLOAD
    if condition > MAX_CONDITION:
        return PARALLEL_POSES
    spread = float(magnitudes.max() - magnitudes.min())
    budget = slip_tolerance * float(magnitudes.mean()) + SLIP_NOISE_MARGIN * force_noise
    if spread > budget:
        return SLIPPED
    return OK


def pose_condition(gravities: Sequence[np.ndarray], mass: float = 1.0) -> float:
    """Conditioning a pose set would give, without measuring anything.

    Useful to pick orientations before moving: lower is better.
    """
    forces = [-mass * np.asarray(gravity, dtype=float).reshape(3) for gravity in gravities]
    return float(np.linalg.cond(np.vstack([-skew(force) for force in forces])))
