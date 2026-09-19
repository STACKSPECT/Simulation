from __future__ import annotations

from dataclasses import dataclass

from .models import Package


@dataclass(frozen=True, slots=True)
class SuctionPlan:
    active_cups: tuple[tuple[float, float], ...]
    capacity_newtons: float
    required_newtons: float
    tool_offset: tuple[float, float] = (0.0, 0.0)

    @property
    def feasible(self) -> bool:
        return bool(self.active_cups) and self.capacity_newtons >= self.required_newtons


@dataclass(frozen=True, slots=True)
class SuctionArray:
    """OnRobot VGP20 electric vacuum gripper.

    Catalogue values: 264 x 184 x 92 mm, 2.55 kg, sixteen 40 mm cups, four vacuum
    channels. Sold for cobot carton palletizing; unused cups are blinded so only the
    footprint that actually covers the box contributes force.

    The array spans the whole housing, so a carton shorter than 264 mm cannot reach the
    outer columns and one shallower than 184 mm cannot reach the outer rows. Centring the
    housing on such a carton wastes the cups that fall past its edges, so `plan` instead
    picks the largest contiguous block of cups that fits and reports the offset that puts
    that block -- not the housing -- over the carton centre.
    """

    name: str = "OnRobot VGP20"
    columns: int = 4
    rows: int = 4
    pitch_x: float = 0.0747
    pitch_y: float = 0.048
    cup_radius: float = 0.020
    cup_force_newtons: float = 50.0
    derating: float = 0.70
    safety_factor: float = 1.5
    body_length: float = 0.264
    body_width: float = 0.184
    flange_to_cup: float = 0.092
    mass: float = 2.55

    def cup_offsets(self) -> tuple[tuple[int, int, float, float], ...]:
        cups: list[tuple[int, int, float, float]] = []
        for column in range(self.columns):
            x = (column - (self.columns - 1) / 2) * self.pitch_x
            for row in range(self.rows):
                y = (row - (self.rows - 1) / 2) * self.pitch_y
                cups.append((column, row, x, y))
        return tuple(cups)

    def _axis_block(self, count: int, pitch: float, span: float) -> tuple[tuple[float, ...], float]:
        """Longest run of cup coordinates that fits across `span`, and where it sits.

        Returns the coordinates in tool frame plus the centre of the run, which is what
        the arm has to cancel out to land the run on the middle of the carton. Ties go to
        the run closest to the tool axis, so the load stays near the flange.
        """
        usable = span / 2 - self.cup_radius
        coordinates = [(index - (count - 1) / 2) * pitch for index in range(count)]
        best: tuple[float, ...] = ()
        best_centre = 0.0
        for start in range(count):
            for end in range(start, count):
                if (coordinates[end] - coordinates[start]) / 2 > usable:
                    continue
                run = tuple(coordinates[start : end + 1])
                centre = (coordinates[start] + coordinates[end]) / 2
                if len(run) > len(best) or (len(run) == len(best) and abs(centre) < abs(best_centre)):
                    best, best_centre = run, centre
        return best, best_centre

    def plan(self, package: Package, yaw: int = 0) -> SuctionPlan:
        width, depth, _ = package.oriented_size(yaw)
        xs, offset_x = self._axis_block(self.columns, self.pitch_x, width)
        ys, offset_y = self._axis_block(self.rows, self.pitch_y, depth)
        cups = tuple((x, y) for x in xs for y in ys)
        capacity = len(cups) * self.cup_force_newtons * self.derating
        required = package.mass * 9.81 * self.safety_factor
        return SuctionPlan(cups, capacity, required, (offset_x, offset_y))
