from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeightMap:
    """Discretized highest surface over a pallet XY footprint.

    `heights` is shaped (ny, nx) with world x along axis 1 and y along axis 0.
    Unobserved cells keep `heights == 0` but `observed == False`.
    """

    origin_xy: tuple[float, float]
    length: float
    width: float
    resolution: float
    heights: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=np.float32)
        self.observed = np.asarray(self.observed, dtype=bool)
        if self.heights.shape != self.observed.shape:
            raise ValueError("heights and observed must have the same shape")
        ny, nx = self.grid_shape
        if self.heights.shape != (ny, nx):
            raise ValueError(
                f"height grid {self.heights.shape} does not match "
                f"pallet {(ny, nx)} at resolution {self.resolution}"
            )

    @property
    def grid_shape(self) -> tuple[int, int]:
        nx = int(round(self.length / self.resolution))
        ny = int(round(self.width / self.resolution))
        return ny, nx

    @property
    def nx(self) -> int:
        return self.grid_shape[1]

    @property
    def ny(self) -> int:
        return self.grid_shape[0]

    @property
    def completeness(self) -> float:
        return float(self.observed.mean()) if self.observed.size else 0.0

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        x = self.origin_xy[0] + (i + 0.5) * self.resolution
        y = self.origin_xy[1] + (j + 0.5) * self.resolution
        return x, y

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.grid_shape
        xs = self.origin_xy[0] + (np.arange(nx, dtype=np.float64) + 0.5) * self.resolution
        ys = self.origin_xy[1] + (np.arange(ny, dtype=np.float64) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)

    def copy(self) -> HeightMap:
        return HeightMap(
            origin_xy=self.origin_xy,
            length=self.length,
            width=self.width,
            resolution=self.resolution,
            heights=self.heights.copy(),
            observed=self.observed.copy(),
        )


def empty_height_map(
    length: float,
    width: float,
    resolution: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> HeightMap:
    """Input: pallet footprint and cell size. Output: zero heights, nothing observed."""
    nx = int(round(length / resolution))
    ny = int(round(width / resolution))
    return HeightMap(
        origin_xy=origin_xy,
        length=length,
        width=width,
        resolution=resolution,
        heights=np.zeros((ny, nx), dtype=np.float32),
        observed=np.zeros((ny, nx), dtype=bool),
    )
