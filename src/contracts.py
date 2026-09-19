"""
El idioma común: lo que se pasan visión, planificación, ejecución y medida.

**Este es el único fichero que leen los cuatro módulos.** Si para que dos de ellos se
entiendan hay que importar algo de la carpeta del otro, el contrato está mal y se
arregla aquí, no con un import cruzado.

Nada de esto sabe que la plataforma existe: los nombres son los del dominio, no los de
ninguna columna. La traducción vive en `src/telemetry.py` y en ningún otro sitio.

Unidades: SI en todo. Metros, kilogramos, segundos, radianes. La única excepción del
proyecto son los `payload` de eventos, que van en mm y grados, y esa conversión ocurre
en la frontera de telemetría.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Lo que produce VISIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """
    Un paquete visto en la estación de recogida de la cinta.

    Es lo que la percepción CREE, no lo que hay: `confidence` y `dims_guess` pueden
    estar mal, y el resto del sistema tiene que aguantarlo. Una lista vacía de
    observaciones con la cinta entregando es `no_detection`, que ahora sí puede pasar.
    """

    package_id: str
    position: np.ndarray            # [x, y, z] del centro, en el mundo
    yaw: float
    dims_guess: tuple[float, float, float]
    confidence: float               # 0..1


@dataclass(frozen=True)
class PackageSpec:
    """
    Lo que se sabe del paquete DESPUÉS de cogerlo y medirlo. Esto es lo que planifica.

    `cog_offset_m` es el desplazamiento del centro de gravedad respecto al centro
    geométrico. Si el gauge lo devuelve siempre a cero, o lo copia del catálogo, la
    medición no está aportando nada: ver AGENTS.md §6.
    """

    package_id: str
    type_name: str
    dims_m: tuple[float, float, float]      # COMPLETAS, no semiejes
    mass_kg: float
    cog_offset_m: np.ndarray                # [dx, dy, dz] desde el centro geométrico

    @property
    def grasp_width(self) -> float:
        """Lo que tienen que abarcar los dedos: el lado horizontal menor."""
        return min(self.dims_m[0], self.dims_m[1])

    @property
    def footprint_area(self) -> float:
        return self.dims_m[0] * self.dims_m[1]


# ─────────────────────────────────────────────────────────────────────────────
# Lo que produce PLANIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Heightmap:
    """
    La altura del montón, celda a celda, en metros SOBRE la cubierta del palé.

    Se MIDE del estado de la escena en cada pasada; no es un contador que el
    planificador va actualizando. A la tercera caja torcida, un contador deja de
    coincidir con la realidad y ya no se recupera.
    """

    cells: np.ndarray               # (ny, nx), metros sobre la cubierta
    origin: tuple[float, float]     # esquina (-x, -y) del palé, en el mundo
    cell_size: float                # lado de la celda, en metros

    @property
    def top(self) -> float:
        """Altura del punto más alto del montón."""
        return float(self.cells.max()) if self.cells.size else 0.0


@dataclass(frozen=True)
class PlacementPlan:
    """
    El hueco elegido para este paquete, y por qué.

    `breakdown` es el desglose del score, término a término. Viaja al `payload` del
    evento `plan` —donde sobrar es inocuo— y es lo que permite responder "¿por qué puso
    esa caja ahí?" sin volver a correr el episodio.
    """

    position: np.ndarray            # [x, y, z] donde debe quedar el CENTRO, en el mundo
    yaw: float
    layer: int
    slot: str                       # nombre del hueco; sale en el feed de la interfaz
    score: float                    # 0..1
    predicted_support: float        # 0..1, fracción de huella que quedaría apoyada
    breakdown: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Lo que produce MEDIDA
#
# `Placement` y `PalletState` se portan tal cual desde
# `~/HackSpain/paletizado-guionizado/src/pallet/measure.py`, junto con el resto del
# módulo. Se declaran allí, no aquí, porque son el resultado de medir y no un contrato
# que haya que negociar: quien los produce es también quien los define.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Las piezas intercambiables
#
# Cada una tiene DOS implementaciones: la buena y un stub-oráculo que lee la verdad de
# la escena. El stub se entrega ANTES, para que el bucle entero corra verde desde el
# primer día y nadie espere a nadie. Cambiar de una a otra es una línea en
# `scripts/palletize.py`, y `oracle` del run es `any(stub en uso)`.
# ─────────────────────────────────────────────────────────────────────────────

class Detector(Protocol):
    """Ve lo que hay parado en la estación de recogida."""

    def observe(self, scene) -> list[Observation]: ...


class Gauge(Protocol):
    """Mide el paquete que el brazo ya tiene agarrado."""

    def measure(self, scene, arm, observation: Observation) -> PackageSpec: ...


class Planner(Protocol):
    """Elige el hueco. `None` = no hay ninguno válido, y eso es un `wrong_placement`."""

    def choose(self, spec: PackageSpec,
               heightmap: Heightmap) -> PlacementPlan | None: ...


class Sink(Protocol):
    """
    Quien quiera enterarse de lo que ocurre MIENTRAS ocurre.

    Existe por la pantalla Live: la plataforma quiere las filas según se producen, no un
    volcado al final. Lo que pasa por aquí son objetos de dominio, no filas.
    """

    def placement(self, index: int, placement) -> None: ...

    def pallet_state(self, index: int, state, drift: float) -> None: ...

    def event(self, row: dict) -> None: ...

    def snapshot(self, shot) -> None: ...
