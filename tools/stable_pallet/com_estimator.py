"""Payload centre-of-mass estimation from a static wrench balance.

With the tare subtracted, what remains at the sensor is the wrench holding the package
alone:

    F_p   = -m_p * g_s
    tau_p = r x F_p = -[F_p]_x r

Each pose contributes three equations but only rank 2: the component of `r` parallel to
gravity produces no moment and is unobservable. The classical fix is to tilt the wrist
so gravity points somewhere else. This cell usually does not: the planner, the tipping
margin and the pallet CoM all read the load in the contact plane, so the unseen
direction is the one nobody uses. One plumb reading then recovers the two horizontal
components exactly, and the geometric centre of the carton stands in for the vertical.

The fit is a truncated SVD, not a ridge. Truncation leaves the observable plane identical
to a full-rank solve; the prior is what to do where there are no data. Tikhonov would
pull the horizontal components toward the prior, which is the opposite of what we want.

The same path serves both precisions: stack several tilted poses into `matrix` and
`torque` and all three directions resolve, so the prior leaves no trace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .tare import TareCalibration
from .wrench import WrenchSample, skew

MIN_POSES = 1
# Two non-parallel poses are what a full-rank (3-D) solve needs.
MIN_FULL_RANK_POSES = 2
# Above this the poses are nearly parallel and the third component of `r` is not
# observable. Only a verdict when the caller asked for more than one pose.
MAX_CONDITION = 1.0e3
# Below this mass the signal cannot be told apart from the tare residual.
MIN_PAYLOAD_MASS = 0.05  # kg
# Relative drift of |F| between poses that already betrays the package having moved.
SLIP_TOLERANCE = 0.02
# Noise sigmas tolerated before calling the |F| spread a slip. Without this margin the
# noise alone exceeds any relative threshold on light payloads and the detector always
# fires.
SLIP_NOISE_MARGIN = 4.0
# Singular values below this fraction of the largest are treated as unseen.
RANK_TOLERANCE = 1e-3

OK = "ok"
NO_PAYLOAD = "no_payload"
PARALLEL_POSES = "parallel_poses"
SLIPPED = "slipped"
TILTED = "tilted"
# Worst-case planar leak of the unseen component that still counts as a plumb reading.
# The planner's tipping margins start at 8 mm; with the tool hanging straight down this
# cell sees about 1.8 mm.
HIDDEN_LIMIT_M = 0.010


@dataclass(frozen=True, slots=True)
class ComEstimate:
    """Estimated centre of mass and how much it can be trusted."""

    com: np.ndarray  # (3,) m, in the sensor frame
    mass: float  # kg
    condition: float  # conditioning of the pose set
    residual: float  # Nm, rms residual of the fit
    slip: float  # relative drift of |F| between poses
    reason: str  # OK, or why the estimate is not trustworthy
    blind_axis: np.ndarray  # (3,) unit vector in the sensor frame
    rank: int  # how many directions the pose set actually saw

    def __post_init__(self) -> None:
        object.__setattr__(self, "com", np.asarray(self.com, dtype=float).reshape(3))
        object.__setattr__(self, "blind_axis", np.asarray(self.blind_axis, dtype=float).reshape(3))

    @property
    def trustworthy(self) -> bool:
        return self.reason == OK


def estimate_com(
    samples: Sequence[WrenchSample],
    tare: TareCalibration,
    *,
    prior: np.ndarray | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
    slip_tolerance: float = SLIP_TOLERANCE,
    force_noise: float = 0.0,
) -> ComEstimate:
    """Solve the package centre of mass from one or more static poses.

    `prior` is where the centre of mass is assumed to sit along directions the pose set
    does not resolve: the geometric centre of the carton, in the sensor frame. Passing
    zeros puts that component at the sensor origin, tens of centimetres above the box.
    """
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
    anchor = np.zeros(3) if prior is None else np.asarray(prior, dtype=float).reshape(3)
    com, blind_axis, rank = _truncated_solve(matrix, target, anchor, rank_tolerance)
    condition = float(np.linalg.cond(matrix))

    return ComEstimate(
        com=com,
        mass=mass,
        condition=condition,
        residual=float(np.sqrt(np.mean(np.square(matrix @ com - target)))),
        slip=slip,
        reason=_verdict(mass, condition, magnitudes, len(samples), slip_tolerance, force_noise),
        blind_axis=blind_axis,
        rank=rank,
    )


def _truncated_solve(
    matrix: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    rank_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Least-squares on the observable subspace; `prior` fills the rest."""
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or float(singular[0]) <= 0.0:
        return prior.copy(), np.array([0.0, 0.0, 1.0]), 0
    resolved = singular > float(singular[0]) * rank_tolerance
    if not np.any(resolved):
        return prior.copy(), right[-1], 0
    delta = target - matrix @ prior
    correction = right[resolved].T @ ((left[:, resolved].T @ delta) / singular[resolved])
    return prior + correction, right[-1], int(np.sum(resolved))


def hidden_planar_error(
    blind_axis: np.ndarray,
    package_from_sensor: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[float, float]:
    """How much of the unseen component could leak into the package's XY plane.

    `package_from_sensor` takes package coordinates into the sensor frame (the grasp
    rotation). Returns `(lean, hidden_m)`: `lean` is the sine of the angle between the
    blind axis and the package vertical, and `hidden_m` is the worst-case planar error
    given that a centre of mass cannot sit outside its own box.
    """
    rotation = np.asarray(package_from_sensor, dtype=float).reshape(3, 3)
    blind = rotation.T @ np.asarray(blind_axis, dtype=float).reshape(3)
    lean = float(np.hypot(blind[0], blind[1]))
    hidden = float(np.abs(blind) @ np.asarray(half_extents, dtype=float).reshape(3)) * lean
    return lean, hidden


def _verdict(
    mass: float,
    condition: float,
    magnitudes: np.ndarray,
    n_poses: int,
    slip_tolerance: float,
    force_noise: float,
) -> str:
    if mass < MIN_PAYLOAD_MASS:
        return NO_PAYLOAD
    if n_poses >= MIN_FULL_RANK_POSES and condition > MAX_CONDITION:
        return PARALLEL_POSES
    if n_poses >= MIN_FULL_RANK_POSES:
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
