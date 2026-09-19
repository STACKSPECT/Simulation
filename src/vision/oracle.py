"""
Los stubs-oráculo de visión: hacen trampa, a propósito y con el nombre puesto.

Leen la verdad directamente de la escena —dónde está cada caja, cuánto pesa, dónde tiene
el CoG— y devuelven el contrato sin error. Existen para que el bucle entero corra verde
desde el primer día: con esto, quien trabaja en la heurística puede probar lo suyo antes
de que exista la percepción, y al revés.

**Se entregan ANTES que las implementaciones de verdad.** Es la regla que evita que los
cuatro módulos se esperen unos a otros.

Y por eso mismo: **un run con estos stubs va con `oracle=True`**. La interfaz no compara
un run con oráculo contra uno sin él, y marcarlo mal invalida justo la comparación que
justifica el trabajo. El cálculo está en `scripts/palletize.py`, no aquí.

Este es el ÚNICO sitio de `src/vision/` donde está permitido importar de `src/cell/` y
mirar el estado del simulador.
"""

from __future__ import annotations

import numpy as np

from src.contracts import Observation, PackageSpec


class OracleDetector:
    """Devuelve sin error el paquete que la fuente tiene presentado."""

    def observe(self, scene) -> list[Observation]:
        current = getattr(scene.supply, "current", None)
        if current is None:
            return []
        box = scene.boxes[current]
        position, _ = scene.box_pose(current)
        return [Observation(
            package_id=box.package_id,
            position=position,
            yaw=scene.box_yaw(current),
            dims_guess=box.dims_m,
            confidence=1.0,
        )]


class OracleGauge:
    """Lee dimensiones, masa y CoG de la verdad interna de la escena."""

    def measure(self, scene, arm, observation: Observation) -> PackageSpec:
        box = next(box for box in scene.boxes if box.package_id == observation.package_id)
        return self._spec(box)

    def forecast(self, scene, package_ids: list[str]) -> list[PackageSpec]:
        """Da al lookahead oráculo las especificaciones de la carga aún no medida."""
        boxes = {box.package_id: box for box in scene.boxes}
        return [self._spec(boxes[package_id]) for package_id in package_ids]

    @staticmethod
    def _spec(box) -> PackageSpec:
        return PackageSpec(
            package_id=box.package_id,
            type_name=box.type_name,
            dims_m=box.dims_m,
            mass_kg=box.mass_kg,
            cog_offset_m=np.asarray(box.cog_offset_m, dtype=float),
        )
