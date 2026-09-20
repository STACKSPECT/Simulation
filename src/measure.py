"""Mide colocaciones y el estado físico del palé.

La geometría procede del simulador guionizado. La diferencia importante es que el
centro de gravedad usa ``PackageSpec.cog_offset_m`` y que el objetivo de cada caja lo
decide el planificador durante el episodio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from theker_telemetry import TRANSPORT_ACCEL_G, stability_margin, support_polygon  # noqa: F401

from src.contracts import PackageSpec, PlacementPlan

Rect = tuple[float, float, float, float]


@dataclass
class Placement:
    """Una colocación intentada, medida después de que la física se asiente."""

    box: Any
    spec: PackageSpec
    plan: PlacementPlan
    position: np.ndarray
    yaw: float
    planned: np.ndarray
    error_xy: float
    error_yaw: float
    support_ratio: float
    overhang: float
    placed: bool

    @property
    def rect(self) -> Rect:
        return footprint(self.position[0], self.position[1], self.spec.dims_m, self.yaw)

    @property
    def top_z(self) -> float:
        return float(self.position[2]) + self.spec.dims_m[2] / 2.0

    @property
    def layer(self) -> int:
        return int(self.plan.layer)


@dataclass
class PalletState:
    """Masa, CoG y estabilidad del montón en el frame del palé."""

    mass_kg: float
    cog: np.ndarray
    stability_margin: float
    fill_ratio: float


def footprint(x: float, y: float, dims, yaw: float) -> Rect:
    """Envolvente alineada de la huella girada; sobreestima de forma pesimista."""
    cosine, sine = abs(np.cos(yaw)), abs(np.sin(yaw))
    return (
        float(x),
        float(y),
        float(dims[0] * cosine + dims[1] * sine),
        float(dims[0] * sine + dims[1] * cosine),
    )


def rect_overlap(a: Rect, b: Rect) -> float:
    """Área de solape entre dos rectángulos alineados, en m²."""
    ax, ay, adx, ady = a
    bx, by, bdx, bdy = b
    dx = min(ax + adx / 2, bx + bdx / 2) - max(ax - adx / 2, bx - bdx / 2)
    dy = min(ay + ady / 2, by + bdy / 2) - max(ay - ady / 2, by - bdy / 2)
    return max(0.0, dx) * max(0.0, dy)


def support_ratio(rect: Rect, supports: list[Rect]) -> float:
    """Fracción de la huella que apoya en la cubierta o en la capa inferior."""
    area = rect[2] * rect[3]
    if area <= 0.0:
        return 0.0
    return min(1.0, sum(rect_overlap(rect, support) for support in supports) / area)


def overhang(rect: Rect, pallet: Rect) -> float:
    """Máxima distancia que la huella sobresale del palé, en metros."""
    x, y, dx, dy = rect
    px, py, pdx, pdy = pallet
    return max(
        0.0,
        (x + dx / 2) - (px + pdx / 2),
        (px - pdx / 2) - (x - dx / 2),
        (y + dy / 2) - (py + pdy / 2),
        (py - pdy / 2) - (y - dy / 2),
    )


def to_pallet_frame(scene, position) -> np.ndarray:
    """Convierte una posición del mundo al frame de la cubierta del palé."""
    center = np.asarray(scene.pallet_center, dtype=float)
    pos = np.asarray(position, dtype=float)
    return np.array([pos[0] - center[0], pos[1] - center[1], pos[2] - scene.deck_z])


def pallet_rect(scene) -> Rect:
    """Huella de la cubierta en el frame del palé."""
    dx, dy = scene.pallet_dims
    return 0.0, 0.0, float(dx), float(dy)


def on_pallet(scene, placements: list[Placement]) -> list[Placement]:
    """Devuelve las cajas cuya huella toca el palé, aunque estén fuera de tolerancia."""
    deck = pallet_rect(scene)
    return [placement for placement in placements if rect_overlap(placement.rect, deck) > 0.0]


def measure_placement(scene, box, index: int, done: list[Placement]) -> Placement:
    """Mide la caja ``index`` contra el plan y la medida asignados al episodio.

    ``src.episode`` registra ``episode_specs`` y ``episode_plans`` en la escena antes
    de mover el brazo. La escena no calcula ninguno de los dos valores.
    """
    spec: PackageSpec = scene.episode_specs[index]
    plan: PlacementPlan = scene.episode_plans[index]
    world_position, _ = scene.box_pose(box.index)
    position = to_pallet_frame(scene, world_position)
    planned = to_pallet_frame(scene, plan.position)
    yaw = scene.box_yaw(box.index)
    rect = footprint(position[0], position[1], spec.dims_m, yaw)
    supports = (
        [pallet_rect(scene)]
        if plan.layer <= 1
        else [placement.rect for placement in done if placement.layer == plan.layer - 1]
    )
    error_xy = float(np.linalg.norm(position[:2] - planned[:2]))
    period = np.pi / 2 if abs(spec.dims_m[0] - spec.dims_m[1]) < 1e-9 else np.pi
    error_yaw = abs((yaw - plan.yaw + period / 2) % period - period / 2)
    protrusion = overhang(rect, pallet_rect(scene))
    limits = scene.cfg["episode"]
    return Placement(
        box=box,
        spec=spec,
        plan=plan,
        position=position,
        yaw=float(yaw),
        planned=planned,
        error_xy=error_xy,
        error_yaw=float(error_yaw),
        support_ratio=support_ratio(rect, supports),
        overhang=protrusion,
        placed=(
            error_xy <= float(limits["tolerance_xy"])
            and error_yaw <= float(limits["tolerance_yaw"])
            and protrusion <= float(limits["max_overhang"])
        ),
    )


def _package_cog(placement: Placement) -> np.ndarray:
    """CoG real medido del paquete, expresado en el frame del palé."""
    dx, dy, dz = np.asarray(placement.spec.cog_offset_m, dtype=float)
    cosine, sine = np.cos(placement.yaw), np.sin(placement.yaw)
    rotated = np.array([cosine * dx - sine * dy, sine * dx + cosine * dy, dz])
    return placement.position + rotated


def pallet_state(placements: list[Placement]) -> PalletState:
    """Calcula el estado con todas las cajas que pesan sobre el palé."""
    if not placements:
        return PalletState(0.0, np.zeros(3), 0.0, 0.0)
    mass = sum(placement.spec.mass_kg for placement in placements)
    cog = sum(
        placement.spec.mass_kg * _package_cog(placement) for placement in placements
    ) / mass
    base = [placement.rect for placement in placements if placement.layer == 1]
    return PalletState(
        mass_kg=float(mass),
        cog=np.asarray(cog, dtype=float),
        stability_margin=stability_margin(
            float(cog[0]), float(cog[1]), float(cog[2]), base
        ),
        fill_ratio=fill_ratio(placements),
    )


def fill_ratio(placements: list[Placement]) -> float:
    """Volumen de carga dividido por el volumen de su envolvente."""
    if not placements:
        return 0.0
    volume = sum(float(np.prod(placement.spec.dims_m)) for placement in placements)
    rects = [placement.rect for placement in placements]
    x0 = min(rect[0] - rect[2] / 2 for rect in rects)
    x1 = max(rect[0] + rect[2] / 2 for rect in rects)
    y0 = min(rect[1] - rect[3] / 2 for rect in rects)
    y1 = max(rect[1] + rect[3] / 2 for rect in rects)
    z1 = max(placement.top_z for placement in placements)
    envelope = (x1 - x0) * (y1 - y0) * max(z1, 1e-9)
    return float(min(1.0, volume / envelope)) if envelope > 0.0 else 0.0


def layer_flatness(placements: list[Placement]) -> float:
    """Rango de alturas de la cara superior de la capa más alta."""
    if not placements:
        return 0.0
    top_layer = max(placement.layer for placement in placements)
    tops = [placement.top_z for placement in placements if placement.layer == top_layer]
    return float(max(tops) - min(tops)) if len(tops) > 1 else 0.0


def positions(scene, boxes) -> dict[str, np.ndarray]:
    """Instantánea de posiciones para medir la deriva de todo el montón."""
    return {box.package_id: scene.box_pose(box.index)[0] for box in boxes}


def max_displacement(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> float:
    """Mayor desplazamiento entre dos instantáneas de las mismas cajas."""
    shared = before.keys() & after.keys()
    if not shared:
        return 0.0
    return float(max(np.linalg.norm(after[key] - before[key]) for key in shared))


def demo() -> None:
    """Comprueba geometría, estabilidad y el uso del CoG descentrado."""
    assert rect_overlap((0, 0, 1, 1), (0.5, 0, 1, 1)) == 0.5
    assert rect_overlap((0, 0, 1, 1), (5, 0, 1, 1)) == 0.0
    assert support_ratio((0, 0, 1, 1), [(0, 0, 1, 1)]) == 1.0
    assert round(overhang((0.06, 0, 0.1, 0.07), (0, 0, 0.21, 0.14)), 6) == 0.005

    class Box:
        package_id = "demo"
        type_name = "demo"

    def placement(x: float, cog_x: float, placed: bool = True) -> Placement:
        spec = PackageSpec("demo", "demo", (0.1, 0.06, 0.05), 1.0,
                           np.array([cog_x, 0.0, 0.0]))
        plan = PlacementPlan(np.array([x, 0.0, 0.025]), 0.0, 1, "demo", 1.0, 1.0)
        pos = np.array([x, 0.0, 0.025])
        return Placement(Box(), spec, plan, pos, 0.0, pos.copy(), 0.0, 0.0,
                         1.0, 0.0, placed)

    good = placement(-0.05, 0.0)
    outside_tolerance = placement(-0.05, 0.02, placed=False)
    state = pallet_state([good, outside_tolerance])
    assert state.mass_kg == 2.0
    assert np.isclose(state.cog[0], -0.04)
    base = [(-0.05, 0.0, 0.10, 0.06), (0.05, 0.0, 0.10, 0.06)]
    centered = stability_margin(0.0, 0.0, 0.05, base)
    assert round(centered, 9) == round(0.03 - TRANSPORT_ACCEL_G * 0.05, 9)
    assert stability_margin(0.0, 0.0, 0.10, base) < centered
    assert stability_margin(0.0, 0.05, 0.05, base) < 0.0
    assert stability_margin(0.0, 0.0, 0.0, []) == 0.0
    assert max_displacement({"a": np.zeros(3)}, {"a": np.array([0.02, 0, 0])}) == 0.02
    print("ok  measure.demo")


if __name__ == "__main__":
    demo()
