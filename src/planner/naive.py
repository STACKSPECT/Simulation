"""Línea base en rejilla y beam search portado del demostrador.

``GridPlanner`` es el stub y marca el run como oráculo. ``BeamPlanner`` usa el modelo
de estabilidad de ``tools/stable_pallet`` y no es un stub. Ambos cumplen el contrato
``Planner``; ``record`` y ``set_forecast`` son extensiones que usa el director para
incorporar el resultado físico y el lookahead cuando la fuente lo conoce.
"""

from __future__ import annotations

import math

import numpy as np

from src.contracts import Heightmap, PackageSpec, PlacementPlan
from tools.stable_pallet.models import (
    Package as ToolPackage,
    Pallet as ToolPallet,
    Placement as ToolPlacement,
    StackState,
)
from tools.stable_pallet.planner import (
    NoFeasiblePlacement,
    PlannerConfig,
    PlannerWeights,
    StablePalletPlanner,
)


def _overlap(a, b) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def measured_heightmap(scene) -> Heightmap:
    """Adaptador temporal mientras ``planner.heightmap`` siga sin implementar.

    Se mantiene aquí porque este fichero es la única pieza de planificación incluida en
    la migración. El mapa se mide de la física en cada ciclo; no es un contador.
    """
    cell = float(scene.cfg["heuristic"]["cell_size"])
    width, depth = scene.pallet_dims
    center_x, center_y = scene.pallet_center
    nx, ny = math.ceil(width / cell), math.ceil(depth / cell)
    origin = (center_x - width / 2, center_y - depth / 2)
    cells = np.zeros((ny, nx), dtype=float)
    measured = []
    for box in scene.boxes:
        position, _ = scene.box_pose(box.index)
        if position[2] < scene.deck_z:
            continue
        measured.append((float(position[2] - box.dims_m[2] / 2), box, position))
    for bottom, box, position in sorted(measured, key=lambda item: item[0]):
        yaw = scene.box_yaw(box.index)
        cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
        span_x = box.dims_m[0] * cosine + box.dims_m[1] * sine
        span_y = box.dims_m[0] * sine + box.dims_m[1] * cosine
        x0 = max(0, math.floor((position[0] - span_x / 2 - origin[0]) / cell))
        x1 = min(nx, math.ceil((position[0] + span_x / 2 - origin[0]) / cell))
        y0 = max(0, math.floor((position[1] - span_y / 2 - origin[1]) / cell))
        y1 = min(ny, math.ceil((position[1] + span_y / 2 - origin[1]) / cell))
        if x0 >= x1 or y0 >= y1:
            continue
        # Una caja de la mesa o en la mano puede proyectarse sobre el borde del palé,
        # pero no forma parte del montón si no apoya en la altura ya medida.
        support = cells[y0:y1, x0:x1]
        if bottom > scene.deck_z + float(support.max(initial=0.0)) + 0.02:
            continue
        top = max(0.0, float(position[2] + box.dims_m[2] / 2 - scene.deck_z))
        cells[y0:y1, x0:x1] = np.maximum(cells[y0:y1, x0:x1], top)
    return Heightmap(cells=cells, origin=origin, cell_size=cell)


class GridPlanner:
    """Primer hueco de una rejilla geométrica, sin score ni estabilidad propia."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.records = []

    def record(self, placement) -> None:
        self.records.append(placement)

    def set_forecast(self, specs: list[PackageSpec]) -> None:
        del specs

    def choose(self, spec: PackageSpec, heightmap: Heightmap) -> PlacementPlan | None:
        del heightmap
        pallet = self.cfg["pallet"]
        px, py = pallet["center"]
        pallet_w, pallet_d = pallet["dims"]
        clearance = float(pallet["clearance"])
        levels = sorted({0.0, *(round(item.top_z, 5) for item in self.records)})

        for yaw in (0.0, math.pi / 2):
            width, depth = (
                spec.dims_m[:2] if yaw == 0.0 else (spec.dims_m[1], spec.dims_m[0])
            )
            for z in levels:
                same = [item for item in self.records if abs(item.top_z - z) < 0.006]
                xs = {0.0, (pallet_w - width) / 2}
                ys = {0.0, (pallet_d - depth) / 2}
                for item in self.records:
                    local_x = item.rect[0] + pallet_w / 2
                    local_y = item.rect[1] + pallet_d / 2
                    xs.update({
                        local_x - item.rect[2] / 2 - width - clearance,
                        local_x + item.rect[2] / 2 + clearance,
                    })
                    ys.update({
                        local_y - item.rect[3] / 2 - depth - clearance,
                        local_y + item.rect[3] / 2 + clearance,
                    })
                pairs = sorted(
                    ((x, y) for x in xs for y in ys),
                    key=lambda pair: (
                        abs(pair[0] + width / 2 - pallet_w / 2)
                        + abs(pair[1] + depth / 2 - pallet_d / 2),
                        pair,
                    ),
                )
                for x, y in pairs:
                        if x < -1e-9 or y < -1e-9:
                            continue
                        rect = (x, y, width, depth)
                        if x + width > pallet_w + 1e-9 or y + depth > pallet_d + 1e-9:
                            continue
                        blocked = False
                        for item in self.records:
                            ix = item.rect[0] + pallet_w / 2 - item.rect[2] / 2
                            iy = item.rect[1] + pallet_d / 2 - item.rect[3] / 2
                            vertical = min(z + spec.dims_m[2], item.top_z) - max(
                                z, item.position[2] - item.spec.dims_m[2] / 2
                            )
                            if vertical > 0.003 and _overlap(rect, (ix, iy, item.rect[2], item.rect[3])):
                                blocked = True
                                break
                            if vertical > 0.003:
                                gap_x = max(ix - (x + width), x - (ix + item.rect[2]))
                                gap_y = max(iy - (y + depth), y - (iy + item.rect[3]))
                                if gap_x + 1e-6 < clearance and gap_y + 1e-6 < clearance:
                                    blocked = True
                                    break
                        if blocked:
                            continue
                        support = 1.0 if z == 0.0 else min(
                            1.0,
                            sum(
                                _overlap(rect, (
                                    item.rect[0] + pallet_w / 2 - item.rect[2] / 2,
                                    item.rect[1] + pallet_d / 2 - item.rect[3] / 2,
                                    item.rect[2], item.rect[3],
                                ))
                                for item in same
                            ) / (width * depth),
                        )
                        if support < float(self.cfg["heuristic"]["min_support"]):
                            continue
                        layer = 1 + sum(level < z - 0.003 for level in levels)
                        return PlacementPlan(
                            position=np.array([
                                px - pallet_w / 2 + x + width / 2,
                                py - pallet_d / 2 + y + depth / 2,
                                float(pallet["deck_thickness"]) + z + spec.dims_m[2] / 2,
                            ]),
                            yaw=yaw,
                            layer=layer,
                            slot=f"grid-L{layer}-{len(self.records) + 1}",
                            score=0.0,
                            predicted_support=float(support),
                            breakdown={},
                        )
        return None


class BeamPlanner:
    """Beam search con lookahead y estabilidad, alimentado por medidas físicas."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        pallet_cfg = cfg["pallet"]
        planner_cfg = cfg["planner"]
        weights = PlannerWeights(**planner_cfg["weights"])
        config = PlannerConfig(
            minimum_support_ratio=float(planner_cfg["minimum_support_ratio"]),
            minimum_tipping_margin=float(planner_cfg["minimum_tipping_margin"]),
            beam_width=int(planner_cfg["beam_width"]),
            lookahead=int(planner_cfg["lookahead"]),
            candidates_per_node=int(planner_cfg["candidates_per_node"]),
            placement_clearance=float(pallet_cfg["clearance"]),
            heightmap_resolution=tuple(planner_cfg["heightmap_resolution"]),
            weights=weights,
        )
        self.engine = StablePalletPlanner(config)
        self.pallet = ToolPallet(
            width=float(pallet_cfg["dims"][0]),
            depth=float(pallet_cfg["dims"][1]),
            max_height=float(pallet_cfg["max_height"]),
            max_mass=float(pallet_cfg["max_mass"]),
        )
        self.state = StackState(self.pallet)
        self.forecast: list[PackageSpec] = []

    @staticmethod
    def _package(spec: PackageSpec) -> ToolPackage:
        return ToolPackage(
            id=spec.package_id,
            size=tuple(float(value) for value in spec.dims_m),
            mass=float(spec.mass_kg),
            com=tuple(float(value) for value in spec.cog_offset_m),
        )

    def set_forecast(self, specs: list[PackageSpec]) -> None:
        self.forecast = list(specs)

    def record(self, measured) -> None:
        """Sustituye el estado esperado por la pose que realmente dejó la física."""
        width, depth = measured.spec.dims_m[:2]
        yaw_deg = int(round(math.degrees(measured.plan.yaw) / 90.0) * 90) % 180
        if yaw_deg == 90:
            width, depth = depth, width
        px, py = self.cfg["pallet"]["dims"]
        x = float(measured.position[0] + px / 2 - width / 2)
        y = float(measured.position[1] + py / 2 - depth / 2)
        z = max(0.0, float(measured.position[2] - measured.spec.dims_m[2] / 2))
        actual = ToolPlacement(self._package(measured.spec), x, y, z, yaw_deg)
        self.state = self.state.with_placement(actual)

    def choose(self, spec: PackageSpec, heightmap: Heightmap) -> PlacementPlan | None:
        del heightmap
        incoming = [self._package(spec), *(self._package(item) for item in self.forecast)]
        try:
            candidate = self.engine.plan_next(self.state, incoming)
        except NoFeasiblePlacement:
            return None
        placement = candidate.placement
        center_x, center_y, center_z = placement.center
        pallet_cfg = self.cfg["pallet"]
        levels = sorted({0.0, *(round(item.z, 5) for item in self.state.placements)})
        layer = 1 + sum(level < placement.z - 0.003 for level in levels)
        interface = next(
            (item for item in candidate.report.interfaces
             if item.package_id == placement.package.id),
            None,
        )
        return PlacementPlan(
            position=np.array([
                float(pallet_cfg["center"][0]) - self.pallet.width / 2 + center_x,
                float(pallet_cfg["center"][1]) - self.pallet.depth / 2 + center_y,
                float(pallet_cfg["deck_thickness"]) + center_z,
            ]),
            yaw=math.radians(placement.yaw),
            layer=layer,
            slot=f"beam-L{layer}-{len(self.state.placements) + 1}",
            score=float(candidate.score),
            predicted_support=(1.0 if interface is None else float(interface.support_ratio)),
            breakdown={name: float(value) for name, value in candidate.components.items()},
        )
