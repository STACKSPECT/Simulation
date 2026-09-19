"""Approximate the stacked pallet's centre of mass from what the robot already knows.

After each placement the cell has three numbers for that package: where it put it on
the pallet, what it weighs, and where its own centre of mass sits. The load CoM is the
mass-weighted average of those points. Nothing here reads the simulator: that is the
whole point of the estimate, so it can later be compared with the physical truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

import numpy as np

from .models import Pallet, Placement, StackState


@dataclass(frozen=True, slots=True)
class PalletCom:
    """Centre of mass of the stacked load, in pallet coordinates.

    Origin is the top-left corner of the pallet deck, with z measured up from the deck.
    The wooden pallet itself is a static geom and is not included: the robot only knows
    the packages it has placed.
    """

    com: tuple[float, float, float]  # m
    mass: float  # kg

    def as_dict(self) -> dict[str, object]:
        return {"com_m": list(self.com), "mass_kg": self.mass}


@dataclass(frozen=True, slots=True)
class PalletComComparison:
    """Estimated load CoM next to the physical one, with the residual."""

    estimated: PalletCom
    truth: PalletCom
    error_m: tuple[float, float, float]  # estimated - truth

    @property
    def error_norm_m(self) -> float:
        return hypot(*self.error_m)

    @property
    def error_xy_m(self) -> float:
        return hypot(self.error_m[0], self.error_m[1])

    def as_dict(self) -> dict[str, object]:
        return {
            "estimated_m": list(self.estimated.com),
            "true_m": list(self.truth.com),
            "error_mm": [value * 1_000 for value in self.error_m],
            "error_norm_mm": self.error_norm_m * 1_000,
            "error_xy_mm": self.error_xy_m * 1_000,
            "estimated_mass_kg": self.estimated.mass,
            "true_mass_kg": self.truth.mass,
        }


def estimate_pallet_com(placements: Sequence[Placement], pallet: Pallet) -> PalletCom:
    """Mass-weighted CoM from commanded pallet coordinates, package mass and package CoM."""
    state = StackState(pallet, list(placements))
    return PalletCom(state.global_com, state.total_mass)


def compare_pallet_com(estimated: PalletCom, truth: PalletCom) -> PalletComComparison:
    error = (
        estimated.com[0] - truth.com[0],
        estimated.com[1] - truth.com[1],
        estimated.com[2] - truth.com[2],
    )
    return PalletComComparison(estimated, truth, error)


def com_from_world_points(
    masses: Sequence[float],
    world_coms: Sequence[np.ndarray],
    origin: tuple[float, float],
    pallet_height: float,
    pallet: Pallet,
) -> PalletCom:
    """Mass-weighted CoM of world-frame points, expressed in pallet coordinates."""
    if not masses or sum(masses) <= 0.0:
        return PalletCom((pallet.width / 2, pallet.depth / 2, 0.0), 0.0)
    total = float(sum(masses))
    moment = np.zeros(3, dtype=float)
    for mass, com in zip(masses, world_coms, strict=True):
        moment += float(mass) * np.asarray(com, dtype=float)
    world = moment / total
    return PalletCom(
        (
            float(world[0] - origin[0]),
            float(world[1] - origin[1]),
            float(world[2] - pallet_height),
        ),
        total,
    )
