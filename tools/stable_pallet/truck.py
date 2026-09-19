"""How a trailer arrives loaded, and the only order in which it can be emptied.

An infeed conveyor hands the cell one carton at a time: singulated, always in the same
place, always the same way round. A trailer hands it the whole load at once. The cartons
were stacked in columns by whoever filled it, the only one the arm may take is the one
with nothing resting on it, and there is no belt to re-present the next one.

`plan_truck_load` builds such a load: random cartons, random orientation, random column,
but stacked the way a person stacks them -- aligned rows across the deck, the biggest
footprint at the bottom of every column, and every carton landing squarely on the one it
rests on. Nothing here knows about MuJoCo; the simulator only writes the poses this
module computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, radians, sin
from random import Random
from typing import Any

from .models import Package

_TOLERANCE = 1e-9
_YAWS = (0, 90)


class TruckLoadTooBig(RuntimeError):
    """The cartons do not fit in the part of the trailer the arm can reach into."""


@dataclass(frozen=True, slots=True)
class TruckBay:
    """The reachable rear corner of a trailer floor, in world coordinates.

    A trailer is far deeper than a fixed arm can reach, so the cell only ever works the
    bay in front of the doors: `origin` is its near-left corner, `width` runs along X
    towards the cab and `depth` along Y. The rest is what the loader was disciplined
    about -- gaps between columns, how high a column may go, how much of a carton has to
    land on the one below it -- plus the sloppiness that survives anyway.

    The default bay is the rectangle the arm can *empty*, which is smaller than the one
    it can point at. Solving the kinematics is not the test: at deck level the corner
    nearest the robot needs the arm folded so tightly that a link stands on its own
    pedestal, so the servos stop about 150 mm short of a pose the solver is happy with.
    Driving the tool across the floor of the bay put that wall at x = +0.12, and the far
    corner out of reach past y = -1.30, so the bay stops short of both.
    """

    origin: tuple[float, float] = (-0.51, -1.28)
    width: float = 0.63
    depth: float = 0.82
    floor_height: float = 0.02
    column_gap: float = 0.04
    max_stack_height: float = 0.86
    max_column_items: int = 5
    # A carton has to land squarely on the one below it, and may be nudged off the
    # column axis to clear the pile next to it -- as long as it still does.
    min_support: float = 0.62
    max_slide: float = 0.09
    position_jitter: float = 0.005
    yaw_jitter_deg: float = 1.5

    def __post_init__(self) -> None:
        positives = (
            ("width", self.width),
            ("depth", self.depth),
            ("floor_height", self.floor_height),
            ("max_stack_height", self.max_stack_height),
        )
        for name, value in positives:
            if value <= 0:
                raise ValueError(f"TruckBay.{name} must be positive")
        if self.column_gap < 0 or self.max_slide < 0:
            raise ValueError("Column gap and slide allowance cannot be negative")
        if not 0 < self.min_support <= 1:
            raise ValueError("min_support is a fraction of the carton footprint")
        if self.max_column_items < 1:
            raise ValueError("A column has to hold at least one carton")
        if self.position_jitter < 0 or self.yaw_jitter_deg < 0:
            raise ValueError("Jitter cannot be negative")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """`(min_x, max_x, min_y, max_y)` of the load floor."""
        return (
            self.origin[0],
            self.origin[0] + self.width,
            self.origin[1],
            self.origin[1] + self.depth,
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.origin[0] + self.width / 2, self.origin[1] + self.depth / 2

    @property
    def ceiling(self) -> float:
        """Highest a carton top may sit, so the tool can still come down onto it."""
        return self.floor_height + self.max_stack_height


@dataclass(frozen=True, slots=True)
class TruckSlot:
    """Where one carton sits in the loaded trailer."""

    index: int
    package_id: str
    center: tuple[float, float, float]
    yaw: float
    footprint: tuple[float, float]
    height: float
    column: int
    level: int

    @property
    def top(self) -> float:
        return self.center[2] + self.height / 2

    @property
    def bottom(self) -> float:
        return self.center[2] - self.height / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "column": self.column,
            "level": self.level,
            "center_m": list(self.center),
            "yaw_deg": self.yaw,
            "footprint_m": list(self.footprint),
            "height_m": self.height,
            "top_m": self.top,
        }


@dataclass(slots=True)
class _Column:
    index: int
    center: tuple[float, float]
    yaw: int
    base: tuple[float, float]
    top_center: tuple[float, float]
    top_footprint: tuple[float, float]
    top_z: float
    items: int


@dataclass(slots=True)
class _Row:
    y: float
    depth: float
    cursor: float


def _footprint_area(package: Package) -> float:
    return package.size[0] * package.size[1]


def _spread(bay: TruckBay, footprint: tuple[float, float]) -> tuple[float, float]:
    """Floor a carton needs once it is stacked by hand rather than by drawing.

    The plan is square to the trailer; the load is not. This is the footprint a carton
    can still cover after it has been nudged and turned by the jitter the loader leaves
    behind, widened again by as much as its neighbour can move the other way. Reserving
    it is what keeps two columns from ending up in the same place.
    """
    tilt = radians(bay.yaw_jitter_deg)
    width, depth = footprint
    pad = 2 * bay.position_jitter
    return (
        width * cos(tilt) + depth * sin(tilt) + pad,
        width * sin(tilt) + depth * cos(tilt) + pad,
    )


def _rect(center: tuple[float, float], spread: tuple[float, float]) -> tuple[float, float, float, float]:
    return (
        center[0] - spread[0] / 2,
        center[0] + spread[0] / 2,
        center[1] - spread[1] / 2,
        center[1] + spread[1] / 2,
    )


def _clashes(
    bay: TruckBay,
    center: tuple[float, float],
    footprint: tuple[float, float],
    bottom: float,
    top: float,
    placed: list[TruckSlot],
) -> bool:
    """Whether a carton put here would share space with the bay walls or another carton."""
    left, right, near, far = _rect(center, _spread(bay, footprint))
    min_x, max_x, min_y, max_y = bay.bounds
    if left < min_x - _TOLERANCE or right > max_x + _TOLERANCE:
        return True
    if near < min_y - _TOLERANCE or far > max_y + _TOLERANCE:
        return True
    for slot in placed:
        if min(top, slot.top) - max(bottom, slot.bottom) <= _TOLERANCE:
            continue
        other = _rect(slot.center[:2], _spread(bay, slot.footprint))
        if min(right, other[1]) - max(left, other[0]) > _TOLERANCE and (
            min(far, other[3]) - max(near, other[2]) > _TOLERANCE
        ):
            return True
    return False


def _target_columns(count: int) -> int:
    """How many piles the loader aims for before it starts stacking higher.

    Two cartons per column keeps the load low and every one of them reachable; the
    floor packer is what decides whether that many columns actually fit.
    """
    return max(1, ceil(count / 2))


def _floor_slot(rows: list[_Row], bay: TruckBay, width: float, depth: float) -> tuple[float, float] | None:
    """Reserve floor for a new column, keeping rows aligned across the bay.

    Rows are filled left to right and opened front to back, which is what a hand-loaded
    trailer looks like. Only the row being filled may grow deeper; an earlier one is
    already boxed in by the row behind it.
    """
    if width > bay.width + _TOLERANCE:
        return None
    for position, row in enumerate(rows):
        last = position == len(rows) - 1
        if row.cursor + width > bay.width + _TOLERANCE:
            continue
        if depth > row.depth + _TOLERANCE and not (last and row.y + depth <= bay.depth + _TOLERANCE):
            continue
        x = row.cursor
        row.cursor += width + bay.column_gap
        row.depth = max(row.depth, depth)
        return x, row.y
    y = 0.0 if not rows else rows[-1].y + rows[-1].depth + bay.column_gap
    if y + depth > bay.depth + _TOLERANCE:
        return None
    rows.append(_Row(y=y, depth=depth, cursor=width + bay.column_gap))
    return 0.0, y


def _open_column(
    rows: list[_Row],
    columns: list[_Column],
    bay: TruckBay,
    package: Package,
    placed: list[TruckSlot],
    rng: Random,
) -> _Column | None:
    """Start a new pile on free floor, turned whichever way still fits.

    Reserving a strip of deck is not enough on its own: a carton further up another pile
    may have been pushed sideways over floor that is still free at deck level, so the
    base of a new pile is checked against what is actually standing there.
    """
    if package.size[2] > bay.max_stack_height + _TOLERANCE:
        return None
    yaws = list(_YAWS)
    rng.shuffle(yaws)
    for yaw in yaws:
        footprint = package.oriented_size(yaw)[:2]
        width, depth = _spread(bay, footprint)
        corner = None
        for _ in range(3):
            corner = _floor_slot(rows, bay, width, depth)
            if corner is None:
                break
            center = (bay.origin[0] + corner[0] + width / 2, bay.origin[1] + corner[1] + depth / 2)
            if not _clashes(
                bay,
                center,
                footprint,
                bay.floor_height,
                bay.floor_height + package.size[2],
                placed,
            ):
                break
            corner = None
        if corner is None:
            continue
        column = _Column(
            index=len(columns),
            center=(
                bay.origin[0] + corner[0] + width / 2,
                bay.origin[1] + corner[1] + depth / 2,
            ),
            yaw=yaw,
            base=footprint,
            top_center=(
                bay.origin[0] + corner[0] + width / 2,
                bay.origin[1] + corner[1] + depth / 2,
            ),
            top_footprint=footprint,
            top_z=bay.floor_height,
            items=0,
        )
        columns.append(column)
        return column
    return None


def _support_ratio(
    carrier_center: tuple[float, float],
    carrier: tuple[float, float],
    center: tuple[float, float],
    footprint: tuple[float, float],
) -> float:
    """Fraction of a carton's base that lands on the carton underneath it."""
    left, right, near, far = _rect(center, footprint)
    other = _rect(carrier_center, carrier)
    overlap_x = max(0.0, min(right, other[1]) - max(left, other[0]))
    overlap_y = max(0.0, min(far, other[3]) - max(near, other[2]))
    return overlap_x * overlap_y / (footprint[0] * footprint[1])


def _slides(bay: TruckBay) -> tuple[tuple[float, float], ...]:
    """Offsets from the pile axis, squarely on top first and sideways only if it must."""
    steps = [0.0]
    step = 0.03
    while steps[-1] + step <= bay.max_slide + _TOLERANCE:
        steps.append(steps[-1] + step)
    offsets = [(0.0, 0.0)]
    for shift in steps[1:]:
        offsets += [(shift, 0.0), (-shift, 0.0), (0.0, shift), (0.0, -shift)]
    return tuple(offsets)


def _seat_on(
    column: _Column, bay: TruckBay, package: Package, placed: list[TruckSlot]
) -> tuple[int, tuple[float, float]] | None:
    """How this carton would sit on this pile: which way round, and pushed over how far.

    The pile's own orientation comes first -- a loader lines a carton up with the one
    below it -- and it is only turned across the pile, or pushed off its axis, when that
    is what it takes to clear the pile next to it while still landing squarely on its
    own.
    """
    if column.items >= bay.max_column_items:
        return None
    for yaw in (column.yaw, 90 - column.yaw):
        width, depth, height = package.oriented_size(yaw)
        if column.top_z + height > bay.ceiling + _TOLERANCE:
            return None
        footprint = (width, depth)
        for dx, dy in _slides(bay):
            center = (column.top_center[0] + dx, column.top_center[1] + dy)
            support = _support_ratio(column.top_center, column.top_footprint, center, footprint)
            if support < bay.min_support:
                continue
            if _clashes(bay, center, footprint, column.top_z, column.top_z + height, placed):
                continue
            return yaw, (dx, dy)
    return None


def _choose_column(
    columns: list[_Column],
    bay: TruckBay,
    package: Package,
    placed: list[TruckSlot],
    rng: Random,
    *,
    spread: bool = False,
) -> tuple[_Column, tuple[int, tuple[float, float]]] | None:
    """Pick which pile this carton goes on: the lowest one it fits, ties at random.

    `spread` is what a retry loosens. Filling the lowest pile first keeps the load even
    and every carton easy to reach, but it is also what strands the last carton when the
    piles end up the same height and none of them has room left for it.
    """
    feasible = [
        (column, seat)
        for column, seat in ((column, _seat_on(column, bay, package, placed)) for column in columns)
        if seat is not None
    ]
    if not feasible:
        return None
    if spread:
        return rng.choice(feasible)
    lowest = min(column.top_z for column, _ in feasible)
    return rng.choice([item for item in feasible if item[0].top_z <= lowest + 0.06])


def _stack_on(
    column: _Column,
    bay: TruckBay,
    package: Package,
    index: int,
    seat: tuple[int, tuple[float, float]],
    rng: Random,
) -> TruckSlot:
    yaw, (dx, dy) = seat
    width, depth, height = package.oriented_size(yaw)
    center = (
        column.top_center[0] + dx + rng.uniform(-bay.position_jitter, bay.position_jitter),
        column.top_center[1] + dy + rng.uniform(-bay.position_jitter, bay.position_jitter),
    )
    slot = TruckSlot(
        index=index,
        package_id=package.id,
        center=(center[0], center[1], column.top_z + height / 2),
        yaw=yaw + rng.uniform(-bay.yaw_jitter_deg, bay.yaw_jitter_deg),
        footprint=(width, depth),
        height=height,
        column=column.index,
        level=column.items,
    )
    column.top_z += height
    column.top_center = center
    column.top_footprint = (width, depth)
    column.items += 1
    return slot


def plan_truck_load(
    packages: tuple[Package, ...] | list[Package],
    bay: TruckBay | None = None,
    seed: int = 0,
    attempts: int = 24,
) -> tuple[TruckSlot, ...]:
    """Stack `packages` into the bay: random, but loaded rather than tipped in.

    Cartons are taken biggest footprint first so no column ever carries a base smaller
    than its load, and each one either opens a new pile on free floor or goes on the
    lowest pile that can take it. Which pile that is decides whether the last carton
    still has room, so a load that will not close is drawn again rather than forced --
    the same thing a loader does when the tail end stops fitting. The returned slots are
    in package order; use `unload_order` for the order the arm may take them out.
    """
    bay = bay or TruckBay()
    packages = list(packages)
    if not packages:
        raise ValueError("A truck load needs at least one package")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    refusal: TruckLoadTooBig | None = None
    for attempt in range(attempts):
        try:
            return _attempt_load(packages, bay, seed, attempt)
        except TruckLoadTooBig as exc:
            refusal = exc
    assert refusal is not None
    raise refusal


def _attempt_load(
    packages: list[Package], bay: TruckBay, seed: int, attempt: int
) -> tuple[TruckSlot, ...]:
    rng = Random(seed if attempt == 0 else seed * 1_000_003 + attempt)
    order = list(range(len(packages)))
    rng.shuffle(order)
    order.sort(key=lambda index: -_footprint_area(packages[index]))

    rows: list[_Row] = []
    columns: list[_Column] = []
    target = _target_columns(len(packages))
    slots: list[TruckSlot] = []
    for index in order:
        package = packages[index]
        choice = None
        if len(columns) < target:
            column = _open_column(rows, columns, bay, package, slots, rng)
            choice = None if column is None else (column, (column.yaw, (0.0, 0.0)))
        if choice is None:
            choice = _choose_column(columns, bay, package, slots, rng, spread=attempt > 0)
        if choice is None:
            column = _open_column(rows, columns, bay, package, slots, rng)
            choice = None if column is None else (column, (column.yaw, (0.0, 0.0)))
        if choice is None:
            raise TruckLoadTooBig(
                f"{package.id} ({package.size[0]:.2f} x {package.size[1]:.2f} x "
                f"{package.size[2]:.2f} m) does not fit the reachable bay: "
                f"{len(columns)} columns already fill {bay.width:.2f} x {bay.depth:.2f} m "
                f"up to {bay.max_stack_height:.2f} m"
            )
        slots.append(_stack_on(choice[0], bay, package, index, choice[1], rng))
    slots.sort(key=lambda slot: slot.index)
    return tuple(slots)


def unload_order(slots: tuple[TruckSlot, ...] | list[TruckSlot]) -> tuple[TruckSlot, ...]:
    """The slots highest top face first: the only order that never digs under a carton.

    A carton resting on another always has the higher top of the two, so taking the
    highest top left in the trailer is always a carton with nothing on it.
    """
    return tuple(sorted(slots, key=lambda slot: (-slot.top, slot.column, slot.index)))


def load_summary(slots: tuple[TruckSlot, ...] | list[TruckSlot], bay: TruckBay) -> dict[str, Any]:
    """What the trailer arrived holding, for the run report."""
    ordered = unload_order(slots)
    columns: dict[int, list[TruckSlot]] = {}
    for slot in slots:
        columns.setdefault(slot.column, []).append(slot)
    return {
        "bay": {
            "origin_m": list(bay.origin),
            "width_m": bay.width,
            "depth_m": bay.depth,
            "floor_height_m": bay.floor_height,
            "max_stack_height_m": bay.max_stack_height,
        },
        "columns": len(columns),
        "tallest_column_m": max((slot.top for slot in slots), default=0.0) - bay.floor_height,
        "planned_pick_order": [slot.package_id for slot in ordered],
        "slots": [slot.as_dict() for slot in slots],
    }
