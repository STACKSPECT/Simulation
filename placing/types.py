from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BoxOrder:
    """Inbound box from CV + CoM (processes 2–3). Do not fill this from simulation.

    Input: `size` full (L, W, H) metres, `mass` kg, `com_local` offset from the
    geometric centre, optional `grasp_xyz` suction point on the top face.
    Output: consumed by `pipeline.score_positions` and `pipeline.place_box`.
    """

    name: str
    size: tuple[float, float, float]
    mass: float = 1.0
    com_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_xyz: tuple[float, float, float] | None = None

    @property
    def length(self) -> float:
        return self.size[0]

    @property
    def width(self) -> float:
        return self.size[1]

    @property
    def height(self) -> float:
        return self.size[2]

    def footprint(self, yaw_deg: float) -> tuple[float, float]:
        """Axis-aligned XY size after a 0/90 degree yaw."""
        if int(round(yaw_deg)) % 180 == 90:
            return self.width, self.length
        return self.length, self.width


@dataclass(frozen=True)
class FeasibilityConfig:
    clearance: float = 0.005
    support_tol: float = 1e-3
    allow_unobserved: bool = False
    min_support_ratio: float = 0.0
    max_stack_height: float = 2.0
    yaws: tuple[float, ...] = (0.0, 90.0)
    robot_xy: tuple[float, float] | None = None
    reach_max: float = 0.0
    reach_min: float = 0.0
    reach_knee: float = 0.25
    reach_falloff: float = 0.65
    transit_margin: float = 0.06
    transit_min_lift: float = 0.12

    def reach_limit(self, pad_z: np.ndarray | float) -> np.ndarray:
        """Horizontal reach available with the suction pad at height `pad_z`.

        A top-down pad loses horizontal reach as it rises above the shoulder,
        so a single scalar radius would accept poses the arm cannot hold.
        """
        above = np.maximum(np.asarray(pad_z, dtype=np.float64) - self.reach_knee, 0.0)
        return np.maximum(self.reach_max - self.reach_falloff * above, 0.0)


@dataclass(frozen=True)
class PalletState:
    """Running stack mass and CoM. Filled by process 7 or `pipeline.next_pallet_state`.

    Input: `mass` kg of boxes already on the pallet, `com_xy` world XY of that
    stack CoM, `pallet_center_xy` / `pallet_half_diag` for the `pallet_com` metric.
    Output: passed into `pipeline.score_positions`. Omit to leave `pallet_com` neutral.
    """

    mass: float = 0.0
    com_xy: tuple[float, float] = (0.0, 0.0)
    pallet_center_xy: tuple[float, float] | None = None
    pallet_half_diag: float = 0.0


@dataclass(frozen=True)
class Placement:
    """Chosen pose for process 6. `x, y` are the box centre; `z_base` is the bottom Z."""

    x: float
    y: float
    z: float
    yaw: float
    z_base: float
    support_ratio: float
    contact_area: float
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class CandidateBatch:
    """All discrete (x, y, yaw) candidates, valid and invalid, as parallel arrays."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    z_base: np.ndarray
    support_ratio: np.ndarray
    void_volume: np.ndarray
    collision: np.ndarray
    unobserved: np.ndarray
    com_supported: np.ndarray
    valid: np.ndarray
    i0: np.ndarray
    j0: np.ndarray
    nx_box: np.ndarray
    ny_box: np.ndarray
    reject_reason: np.ndarray

    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    def take(self, index: int) -> dict[str, float | int | str | bool]:
        return {
            "x": float(self.x[index]),
            "y": float(self.y[index]),
            "z_base": float(self.z_base[index]),
            "yaw": float(self.yaw[index]),
            "support_ratio": float(self.support_ratio[index]),
            "void_volume": float(self.void_volume[index]),
            "collision": bool(self.collision[index]),
            "unobserved": bool(self.unobserved[index]),
            "com_supported": bool(self.com_supported[index]),
            "valid": bool(self.valid[index]),
            "i0": int(self.i0[index]),
            "j0": int(self.j0[index]),
            "nx_box": int(self.nx_box[index]),
            "ny_box": int(self.ny_box[index]),
            "reject_reason": str(self.reject_reason[index]),
        }


def empty_batch() -> CandidateBatch:
    zf = np.zeros(0, dtype=np.float64)
    zi = np.zeros(0, dtype=np.int32)
    zb = np.zeros(0, dtype=bool)
    return CandidateBatch(
        x=zf.copy(),
        y=zf.copy(),
        yaw=zf.copy(),
        z_base=zf.copy(),
        support_ratio=zf.copy(),
        void_volume=zf.copy(),
        collision=zb.copy(),
        unobserved=zb.copy(),
        com_supported=zb.copy(),
        valid=zb.copy(),
        i0=zi.copy(),
        j0=zi.copy(),
        nx_box=zi.copy(),
        ny_box=zi.copy(),
        reject_reason=np.zeros(0, dtype=object),
    )


@dataclass
class PlanResult:
    """Output of process 5. `best` is None when every candidate was rejected."""

    best: Placement | None
    batch: CandidateBatch
    scores: np.ndarray
    metrics: dict[str, np.ndarray] = field(default_factory=dict)
    n_valid: int = 0
    # The grid the candidate indices refer to. The planner resamples the input
    # map, so anything plotting `batch.i0`/`batch.j0` must use this one.
    height_map: object | None = None
