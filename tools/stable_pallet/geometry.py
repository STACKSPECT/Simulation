from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot

EPSILON = 1e-7


@dataclass(frozen=True, slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def corners(self) -> list[tuple[float, float]]:
        return [(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)]

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def intersection(self, other: Rect) -> Rect | None:
        overlap = Rect(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )
        return overlap if overlap.area > EPSILON else None


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= EPSILON:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= EPSILON:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def signed_polygon_margin(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
    """Return the shortest signed edge distance for a CCW convex polygon."""
    if len(polygon) < 3:
        return float("-inf")
    margins: list[float] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        length = hypot(edge_x, edge_y)
        if length <= EPSILON:
            continue
        margins.append((edge_x * (point[1] - start[1]) - edge_y * (point[0] - start[0])) / length)
    return min(margins, default=float("-inf"))
