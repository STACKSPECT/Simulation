from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, radians, sin
from typing import Any


@dataclass(frozen=True, slots=True)
class Pallet:
    width: float
    depth: float
    max_height: float = 1.8
    max_mass: float = 1_000.0

    def __post_init__(self) -> None:
        if min(self.width, self.depth, self.max_height, self.max_mass) <= 0:
            raise ValueError("Pallet dimensions and limits must be positive")


@dataclass(frozen=True, slots=True)
class Package:
    id: str
    size: tuple[float, float, float]
    mass: float
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 0.8

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Package id cannot be empty")
        if len(self.size) != 3 or min(self.size) <= 0:
            raise ValueError(f"Invalid dimensions for package {self.id}")
        if self.mass <= 0:
            raise ValueError(f"Invalid mass for package {self.id}")
        if any(
            abs(offset) > dimension / 2
            for offset, dimension in zip(self.com, self.size, strict=True)
        ):
            raise ValueError(f"Center of mass is outside package {self.id}")

    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def density(self) -> float:
        return self.mass / self.volume

    def oriented_size(self, yaw: int) -> tuple[float, float, float]:
        if yaw % 180 == 0:
            return self.size
        if yaw % 180 == 90:
            return self.size[1], self.size[0], self.size[2]
        raise ValueError("Only 0 and 90 degree placements are supported")

    def rotated_com_xy(self, yaw: int) -> tuple[float, float]:
        angle = radians(yaw)
        dx, dy, _ = self.com
        return cos(angle) * dx - sin(angle) * dy, sin(angle) * dx + cos(angle) * dy


@dataclass(frozen=True, slots=True)
class Placement:
    package: Package
    x: float
    y: float
    z: float
    yaw: int = 0

    @property
    def width(self) -> float:
        return self.package.oriented_size(self.yaw)[0]

    @property
    def depth(self) -> float:
        return self.package.oriented_size(self.yaw)[1]

    @property
    def height(self) -> float:
        return self.package.size[2]

    @property
    def top(self) -> float:
        return self.z + self.height

    @property
    def center(self) -> tuple[float, float, float]:
        return self.x + self.width / 2, self.y + self.depth / 2, self.z + self.height / 2

    @property
    def com_world(self) -> tuple[float, float, float]:
        dx, dy = self.package.rotated_com_xy(self.yaw)
        cx, cy, cz = self.center
        return cx + dx, cy + dy, cz + self.package.com[2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "size": list(self.package.oriented_size(self.yaw)),
            "mass": self.package.mass,
        }


@dataclass(slots=True)
class StackState:
    pallet: Pallet
    placements: list[Placement] = field(default_factory=list)

    @property
    def total_mass(self) -> float:
        return sum(item.package.mass for item in self.placements)

    @property
    def max_height(self) -> float:
        return max((item.top for item in self.placements), default=0.0)

    @property
    def global_com(self) -> tuple[float, float, float]:
        mass = self.total_mass
        if mass == 0:
            return self.pallet.width / 2, self.pallet.depth / 2, 0.0
        moments = [0.0, 0.0, 0.0]
        for placement in self.placements:
            for axis, coordinate in enumerate(placement.com_world):
                moments[axis] += coordinate * placement.package.mass
        return tuple(moment / mass for moment in moments)  # type: ignore[return-value]

    def with_placement(self, placement: Placement) -> StackState:
        return StackState(self.pallet, [*self.placements, placement])

    def as_dict(self) -> dict[str, Any]:
        return {
            "pallet": asdict(self.pallet),
            "total_mass": self.total_mass,
            "max_height": self.max_height,
            "global_com": list(self.global_com),
            "placements": [placement.as_dict() for placement in self.placements],
        }
