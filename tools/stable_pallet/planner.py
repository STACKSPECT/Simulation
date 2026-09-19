from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from .models import Package, Placement, StackState
from .stability import StabilityReport, footprint, has_top_down_access, overlaps_3d, validate_stack


@dataclass(frozen=True, slots=True)
class PlannerWeights:
    support: float = 3.5
    tipping_margin: float = 2.5
    centered_com: float = 2.0
    low_height: float = 1.2
    compactness: float = 1.0
    regular_surface: float = 0.8


@dataclass(frozen=True, slots=True)
class Candidate:
    placement: Placement
    score: float
    report: StabilityReport
    components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerConfig:
    minimum_support_ratio: float = 0.60
    minimum_tipping_margin: float = 0.008
    beam_width: int = 8
    lookahead: int = 3
    candidates_per_node: int = 32
    placement_clearance: float = 0.012
    heightmap_resolution: tuple[int, int] = (12, 10)
    weights: PlannerWeights = field(default_factory=PlannerWeights)


class NoFeasiblePlacement(RuntimeError):
    pass


class StablePalletPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan_next(self, state: StackState, incoming: list[Package]) -> Candidate:
        if not incoming:
            raise ValueError("At least one incoming package is required")

        depth = min(self.config.lookahead, len(incoming))
        # (state, moves, accumulated utility)
        beam: list[tuple[StackState, list[Candidate], float]] = [(state, [], 0.0)]
        for package in incoming[:depth]:
            expanded: list[tuple[StackState, list[Candidate], float]] = []
            for node_state, moves, utility in beam:
                for candidate in self.enumerate_candidates(node_state, package):
                    expanded.append(
                        (
                            node_state.with_placement(candidate.placement),
                            [*moves, candidate],
                            utility + candidate.score,
                        )
                    )
            if not expanded:
                if beam and beam[0][1]:
                    break
                raise NoFeasiblePlacement(f"No stable placement found for {package.id}")
            expanded.sort(key=lambda item: item[2], reverse=True)
            beam = expanded[: self.config.beam_width]

        if not beam or not beam[0][1]:
            raise NoFeasiblePlacement(f"No stable placement found for {incoming[0].id}")
        beam.sort(key=lambda item: item[2], reverse=True)
        return beam[0][1][0]

    def enumerate_candidates(self, state: StackState, package: Package) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: set[tuple[int, int, int, int]] = set()
        z_levels = sorted({0.0, *(round(item.top, 6) for item in state.placements)})

        for yaw in (0, 90):
            width, depth, _ = package.oriented_size(yaw)
            if width > state.pallet.width or depth > state.pallet.depth:
                continue
            clearance = self.config.placement_clearance
            edge_margin = clearance / 2
            xs = {
                edge_margin,
                (state.pallet.width - width) / 2,
                state.pallet.width - width - edge_margin,
            }
            ys = {
                edge_margin,
                (state.pallet.depth - depth) / 2,
                state.pallet.depth - depth - edge_margin,
            }
            for placed in state.placements:
                rect = footprint(placed)
                xs.update(
                    (
                        rect.x0 - width - clearance,
                        rect.x0,
                        rect.x1 - width,
                        rect.x1 + clearance,
                    )
                )
                ys.update(
                    (
                        rect.y0 - depth - clearance,
                        rect.y0,
                        rect.y1 - depth,
                        rect.y1 + clearance,
                    )
                )

            valid_x = sorted(
                x
                for x in xs
                if edge_margin - 1e-7 <= x <= state.pallet.width - width - edge_margin + 1e-7
            )
            valid_y = sorted(
                y
                for y in ys
                if edge_margin - 1e-7 <= y <= state.pallet.depth - depth - edge_margin + 1e-7
            )
            for z in z_levels:
                for x in valid_x:
                    for y in valid_y:
                        key = (round(x * 10_000), round(y * 10_000), round(z * 10_000), yaw)
                        if key in seen:
                            continue
                        seen.add(key)
                        placement = Placement(package, max(0.0, x), max(0.0, y), z, yaw)
                        # Measured stacks need millimetric contact tolerance, but a new
                        # command must neither intersect them nor require insertion below
                        # an overhang: the vacuum tool always approaches from above.
                        if any(overlaps_3d(placement, item, tolerance=1e-6) for item in state.placements):
                            continue
                        if any(
                            self._too_close_on_same_layer(placement, item, clearance)
                            for item in state.placements
                        ):
                            continue
                        if not has_top_down_access(state, placement):
                            continue
                        candidate_state = state.with_placement(placement)
                        report = validate_stack(
                            candidate_state,
                            self.config.minimum_support_ratio,
                            self.config.minimum_tipping_margin,
                        )
                        if not report.stable:
                            continue
                        components = self._score_components(candidate_state, report)
                        score = sum(
                            components[name] * getattr(self.config.weights, name)
                            for name in components
                        )
                        candidates.append(Candidate(placement, score, report, components))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[: self.config.candidates_per_node]

    @staticmethod
    def _too_close_on_same_layer(
        candidate: Placement, existing: Placement, clearance: float
    ) -> bool:
        vertical_overlap = min(candidate.top, existing.top) - max(candidate.z, existing.z)
        if vertical_overlap <= 1e-6:
            return False
        candidate_rect, existing_rect = footprint(candidate), footprint(existing)
        gap_x = max(existing_rect.x0 - candidate_rect.x1, candidate_rect.x0 - existing_rect.x1)
        gap_y = max(existing_rect.y0 - candidate_rect.y1, candidate_rect.y0 - existing_rect.y1)
        return gap_x < clearance and gap_y < clearance

    def _score_components(
        self, state: StackState, report: StabilityReport
    ) -> dict[str, float]:
        newest = state.placements[-1]
        interface = next(item for item in report.interfaces if item.package_id == newest.package.id)
        support = interface.support_ratio
        margin_scale = max(min(newest.width, newest.depth) / 2, 1e-6)
        tipping_margin = min(1.0, max(0.0, interface.margin / margin_scale))

        com_x, com_y, _ = state.global_com
        center_x, center_y = state.pallet.width / 2, state.pallet.depth / 2
        max_distance = hypot(center_x, center_y)
        centered_com = 1.0 - min(1.0, hypot(com_x - center_x, com_y - center_y) / max_distance)
        low_height = 1.0 - min(1.0, state.max_height / state.pallet.max_height)

        volume = sum(
            item.width * item.depth * item.height for item in state.placements
        )
        envelope = state.pallet.width * state.pallet.depth * max(state.max_height, 1e-6)
        compactness = min(1.0, volume / envelope)
        regular_surface = self._surface_regularity(state)
        return {
            "support": support,
            "tipping_margin": tipping_margin,
            "centered_com": centered_com,
            "low_height": low_height,
            "compactness": compactness,
            "regular_surface": regular_surface,
        }

    def _surface_regularity(self, state: StackState) -> float:
        cells_x, cells_y = self.config.heightmap_resolution
        heights: list[float] = []
        for ix in range(cells_x):
            x = state.pallet.width * (ix + 0.5) / cells_x
            for iy in range(cells_y):
                y = state.pallet.depth * (iy + 0.5) / cells_y
                height = max(
                    (
                        placement.top
                        for placement in state.placements
                        if placement.x <= x <= placement.x + placement.width
                        and placement.y <= y <= placement.y + placement.depth
                    ),
                    default=0.0,
                )
                heights.append(height)
        mean = sum(heights) / len(heights)
        variance = sum((height - mean) ** 2 for height in heights) / len(heights)
        return 1.0 - min(1.0, variance**0.5 / max(state.pallet.max_height, 1e-6))
