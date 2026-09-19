from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import EPSILON, Rect, convex_hull, signed_polygon_margin
from .models import Placement, StackState


@dataclass(frozen=True, slots=True)
class Contact:
    supporter_index: int | None
    area: Rect


@dataclass(frozen=True, slots=True)
class InterfaceResult:
    package_id: str
    support_ratio: float
    margin: float
    resultant_xy: tuple[float, float]
    load_mass: float
    supporters: tuple[str, ...]


@dataclass(slots=True)
class StabilityReport:
    stable: bool
    reason: str = ""
    interfaces: list[InterfaceResult] = field(default_factory=list)

    @property
    def minimum_support_ratio(self) -> float:
        return min((item.support_ratio for item in self.interfaces), default=1.0)

    @property
    def minimum_margin(self) -> float:
        return min((item.margin for item in self.interfaces), default=0.0)


def footprint(placement: Placement) -> Rect:
    return Rect(
        placement.x,
        placement.y,
        placement.x + placement.width,
        placement.y + placement.depth,
    )


def overlaps_3d(first: Placement, second: Placement, tolerance: float = 0.003) -> bool:
    first_rect, second_rect = footprint(first), footprint(second)
    overlap_x = min(first_rect.x1, second_rect.x1) - max(first_rect.x0, second_rect.x0)
    overlap_y = min(first_rect.y1, second_rect.y1) - max(first_rect.y0, second_rect.y0)
    horizontal = overlap_x > tolerance and overlap_y > tolerance
    vertical = min(first.top, second.top) - max(first.z, second.z) > tolerance
    return horizontal and vertical


def has_top_down_access(state: StackState, placement: Placement, tolerance: float = 0.002) -> bool:
    """Check that a suction-held box can descend vertically into its final pose."""
    candidate_footprint = footprint(placement)
    for existing in state.placements:
        if candidate_footprint.intersection(footprint(existing)) is None:
            continue
        # A horizontally overlapping object may only be below the new box. Any
        # object at or above its final top blocks the vertical insertion corridor.
        if existing.top > placement.z + tolerance:
            return False
    return True


def contacts_below(state: StackState, index: int, tolerance: float = 0.004) -> list[Contact]:
    placement = state.placements[index]
    box_footprint = footprint(placement)
    if placement.z <= tolerance:
        return [Contact(None, box_footprint)]

    contacts: list[Contact] = []
    for supporter_index, supporter in enumerate(state.placements):
        if supporter_index == index or abs(supporter.top - placement.z) > tolerance:
            continue
        overlap = box_footprint.intersection(footprint(supporter))
        if overlap is not None:
            contacts.append(Contact(supporter_index, overlap))
    return contacts


def validate_stack(
    state: StackState,
    minimum_support_ratio: float = 0.55,
    minimum_margin: float = 0.003,
) -> StabilityReport:
    """Check collision, support and quasi-static load transfer from top to bottom.

    Loads arriving from upper boxes are accumulated as point-force moments. At each
    interface the resultant must remain inside the convex hull of the actual contact
    patches. The load is then distributed to lower supporters by contact area.
    """
    pallet = state.pallet
    if state.total_mass > pallet.max_mass + EPSILON:
        return StabilityReport(False, "pallet mass limit exceeded")

    for index, placement in enumerate(state.placements):
        if (
            placement.x < -EPSILON
            or placement.y < -EPSILON
            or placement.x + placement.width > pallet.width + EPSILON
            or placement.y + placement.depth > pallet.depth + EPSILON
            or placement.top > pallet.max_height + EPSILON
        ):
            return StabilityReport(False, f"{placement.package.id} is outside pallet bounds")
        for other in state.placements[index + 1 :]:
            if overlaps_3d(placement, other):
                return StabilityReport(
                    False, f"{placement.package.id} overlaps {other.package.id}"
                )

    count = len(state.placements)
    load_mass = [placement.package.mass for placement in state.placements]
    load_moment_x = [
        placement.package.mass * placement.com_world[0] for placement in state.placements
    ]
    load_moment_y = [
        placement.package.mass * placement.com_world[1] for placement in state.placements
    ]
    report = StabilityReport(True)

    order = sorted(range(count), key=lambda i: state.placements[i].z, reverse=True)
    for index in order:
        placement = state.placements[index]
        contacts = contacts_below(state, index)
        if not contacts:
            return StabilityReport(False, f"{placement.package.id} has no support", report.interfaces)

        contact_area = sum(contact.area.area for contact in contacts)
        support_ratio = min(1.0, contact_area / (placement.width * placement.depth))
        if support_ratio + EPSILON < minimum_support_ratio:
            return StabilityReport(
                False,
                f"{placement.package.id} support ratio {support_ratio:.3f} is too low",
                report.interfaces,
            )

        polygon = convex_hull(point for contact in contacts for point in contact.area.corners)
        resultant = (
            load_moment_x[index] / load_mass[index],
            load_moment_y[index] / load_mass[index],
        )
        margin = signed_polygon_margin(resultant, polygon)
        if margin + EPSILON < minimum_margin:
            return StabilityReport(
                False,
                f"{placement.package.id} tipping margin {margin:.4f} m is too low",
                report.interfaces,
            )

        supporter_names = tuple(
            "pallet"
            if contact.supporter_index is None
            else state.placements[contact.supporter_index].package.id
            for contact in contacts
        )
        report.interfaces.append(
            InterfaceResult(
                placement.package.id,
                support_ratio,
                margin,
                resultant,
                load_mass[index],
                supporter_names,
            )
        )

        # The pallet is an infinite-capacity sink. Otherwise pass the force through
        # each real contact at that contact's centroid.
        real_contacts = [contact for contact in contacts if contact.supporter_index is not None]
        real_area = sum(contact.area.area for contact in real_contacts)
        for contact in real_contacts:
            assert contact.supporter_index is not None
            fraction = contact.area.area / real_area
            transferred_mass = load_mass[index] * fraction
            cx, cy = contact.area.centroid
            load_mass[contact.supporter_index] += transferred_mass
            load_moment_x[contact.supporter_index] += transferred_mass * cx
            load_moment_y[contact.supporter_index] += transferred_mass * cy

    return report
