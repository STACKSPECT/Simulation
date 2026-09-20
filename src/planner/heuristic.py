"""
La heurística: puntuar cada hueco candidato y quedarse con el mejor.

Las cuentas están en `placing/`, que es numpy y nada más: enumera todas las poses
discretas que caben, descarta las inviables con filtros duros y puntúa el resto con 14
términos, cada uno en [0,1] y combinados en media ponderada. Aquí sólo vive el
**adaptador**: traducir el contrato de este repo al suyo y al revés.

Esa frontera no es cosmética. `placing` no sabe que existe un simulador, ni una cámara,
ni esta aplicación, y por eso los pesos se pueden ajustar y la heurística probar sin
arrancar MuJoCo: un ciclo de tuning pasa de minutos a milisegundos. `python -m placing`
corre 15 comprobaciones, y la primera es que nadie ha colado un import pesado.

── LAS CUATRO TRADUCCIONES, QUE FALLAN EN SILENCIO ──────────────────────────────

Ninguna peta si se olvida: las cajas simplemente acaban donde no toca.

  1. **`yaw` va en GRADOS** en `placing` y en radianes aquí. Cajas giradas 57° = un
     radián tratado como grado.
  2. **Las alturas van en Z DEL MUNDO** en `placing` y sobre la cubierta aquí. Hay que
     desplazar el mapa **y** `max_stack_height`, los dos, y con el mismo convenio. Medio
     convenio funciona: si sólo traduces el mapa, el límite de altura se aplica
     desplazado 144 mm y no lo notas hasta que el montón se pasa de alto o deja de
     aceptar cajas sin motivo.
  3. **`observed`** es del mapa medido; si viene `None` —el oráculo— es todo `True` y
     `allow_unobserved` deja de importar.
  4. **`layer` y `slot` no existen** en `placing`: se derivan aquí.

── LO DEMÁS ─────────────────────────────────────────────────────────────────────

**El score devuelve su desglose.** Los 14 términos van a `PlacementPlan.breakdown` y de
ahí al `payload` del evento `plan`, donde sobrar es inocuo. Es lo que permite responder
"¿por qué puso esa caja ahí?" sin volver a correr el episodio.

**`None` no es una excepción.** Que no quepa en ningún sitio es un resultado normal y se
registra como `wrong_placement`; lanzar desde aquí tumbaría el episodio entero y dejaría
la fila en `running`. El porqué queda en `last_reject`, que es la diferencia entre "no
cupo" y "no cupo porque el brazo ya no llega": la primera cierra el palé, la segunda dice
que el palé está mal colocado respecto al robot.

**El CoG del paquete es un término, no un filtro** (`com_margin`), y **el del palé
también cuenta** (`pallet_com`). Este último va con peso 0.0 por defecto, pero el
`PalletState` que necesita ya se acumula en `record()`: subirlo es editar un número en
`configs/pallet.yaml` y medir, no escribir código.
"""

from __future__ import annotations

import numpy as np

from placing import (
    BoxOrder,
    FeasibilityConfig,
    HeightMap,
    PalletState,
    PlacementPlanner,
    ScoringConfig,
    ranked_placements,
    reject_summary,
)
from placing.scoring import pick_scored_index
from src.contracts import Heightmap, PackageSpec, PlacementPlan

# Dos alturas de base separadas por menos de esto son la misma capa. El mismo umbral que
# usa `GridPlanner`: por debajo de 3 mm lo que hay es deriva al soltar, no una capa nueva.
LAYER_TOL = 0.003


class ScorePlanner:
    """Heurística de score sobre `placing`. Cumple `contracts.Planner`."""

    def __init__(self, pallet_cfg: dict):
        self.cfg = pallet_cfg
        cfg = pallet_cfg["heuristic"]
        pallet = pallet_cfg["pallet"]

        self.deck_z = float(pallet["deck_thickness"])
        length, width = (float(v) for v in pallet["dims"])
        center = tuple(float(v) for v in pallet["center"])

        self.planner = PlacementPlanner(
            feasibility=FeasibilityConfig(
                clearance=float(cfg["clearance"]),
                support_tol=float(cfg["support_tol"]),
                allow_unobserved=bool(cfg["allow_unobserved"]),
                min_support_ratio=float(cfg["min_support_ratio"]),
                # Z DEL MUNDO, igual que el mapa que se le pasa. Ver traducción 2.
                # Sale de `heuristic.max_stack_height`, que es lo que esta celda alcanza
                # de verdad, no de `pallet.max_height`, que es el papel del palé. La
                # cifra escala además `lowness`: el porqué, en `configs/pallet.yaml`.
                max_stack_height=self.deck_z + float(cfg["max_stack_height"]),
                yaws=tuple(float(v) for v in cfg["yaws"]),
                reach_min=float(cfg["reach_min"]),
                reach_knee=float(cfg["reach_knee"]),
                reach_falloff=float(cfg["reach_falloff"]),
            ),
            scoring=ScoringConfig(
                weights={str(k): float(v) for k, v in cfg["weights"].items()},
                proximity_band=float(cfg["proximity_band"]),
                levelness_scale=float(cfg["levelness_scale"]),
                useful_gap=float(cfg["useful_gap"]),
            ),
            # El mapa ya llega con la celda de `cell_size`: no hay nada que remuestrear.
            resolution=None,
            # El alcance va en el planner, no sólo en `FeasibilityConfig`: `placing` sólo
            # convierte `reach_max` en filtro duro cuando el planner trae `robot_xy`.
            # Ponerlo únicamente en la config deja `reach_max` a 0 y el filtro apagado en
            # silencio. Las cifras están medidas en esta celda: ver `configs/pallet.yaml`.
            robot_xy=(tuple(float(v) for v in cfg["robot_xy"])
                      if cfg.get("robot_xy") and float(cfg["reach_max"]) > 0 else None),
            max_reach=float(cfg["reach_max"]),
        )

        # El CoG del montón, acumulado de lo que la física dejó, no de lo planificado.
        self._empty_state = PalletState(
            pallet_center_xy=center,
            pallet_half_diag=0.5 * float(np.hypot(length, width)),
        )
        self.state = self._empty_state
        self._levels: list[float] = []
        self.last_reject: dict[str, int] | None = None
        # Huecos de repuesto del bulto que se está colocando ahora. Ver `alternative`.
        self._retries: list[PlacementPlan] = []

    # ── el contrato ──────────────────────────────────────────────────────────

    def choose(self, spec: PackageSpec,
               heightmap: Heightmap) -> PlacementPlan | None:
        ny, nx = heightmap.cells.shape
        hmap = HeightMap(
            origin_xy=tuple(heightmap.origin),
            length=nx * heightmap.cell_size,
            width=ny * heightmap.cell_size,
            resolution=heightmap.cell_size,
            heights=heightmap.cells + self.deck_z,          # traducción 2
            observed=(                                      # traducción 3
                heightmap.observed
                if heightmap.observed is not None
                else np.ones((ny, nx), dtype=bool)
            ),
        )
        box = BoxOrder(
            name=spec.package_id,
            size=tuple(float(v) for v in spec.dims_m),
            mass=float(spec.mass_kg),
            com_local=tuple(float(v) for v in spec.cog_offset_m),
        )

        result = self.planner.plan(hmap, box, pallet_state=self.state)
        if result.best is None:
            # `{}` no es lo mismo que "todos rechazados": significa que no se generó ni
            # un candidato, o sea que la caja no cabe ni en un palé vacío.
            self.last_reject = reject_summary(result.batch)
            self._retries = []
            return None

        self.last_reject = None
        # La cola de repuesto para ESTE bulto. `ranked_placements` separa los huecos
        # `min_separation` entre sí a propósito: dos celdas vecinas puntúan casi igual, y
        # reintentar un centímetro más allá vuelve a fallar exactamente igual.
        self._retries = [
            self._plan_from(placement, result)
            for placement in ranked_placements(result, box)[1:]
        ]
        return self._plan_from(result.best, result)

    def alternative(self) -> PlacementPlan | None:
        """El siguiente hueco de la cola, cuando el brazo no pudo con el anterior.

        NO se replanifica, y ésa es la razón de que la cola se guarde: no se llegó a
        soltar nada, así que el mapa de alturas es el mismo y `choose` devolvería el
        mismo hueco y el mismo fallo. Lo que ha cambiado no está en el mapa —está en que
        ese hueco ya se sabe que el brazo no lo alcanza—, y eso el mapa no puede verlo.
        """
        return self._retries.pop(0) if self._retries else None

    def _plan_from(self, placement, result) -> PlacementPlan:
        layer = self._layer(placement.z_base)
        return PlacementPlan(
            position=np.array([placement.x, placement.y, placement.z]),  # `z` es el centro
            yaw=float(np.deg2rad(placement.yaw)),           # traducción 1
            layer=layer,
            slot=self._slot(result, layer),                 # traducción 4
            score=float(placement.score),
            predicted_support=float(placement.support_ratio),
            breakdown={name: float(value) for name, value in placement.metrics.items()},
        )

    # ── las extensiones que usa el director ──────────────────────────────────

    def record(self, measured) -> None:
        """Sustituye lo planificado por lo que la física dejó de verdad.

        OJO CON EL FRAME. `measure.Placement.position` NO está en el mundo: viene de
        `measure.to_pallet_frame`, o sea XY respecto al CENTRO del palé y Z sobre la
        CUBIERTA. Tratarlo como mundo no da ningún error —son números plausibles— pero
        deja los niveles 144 mm por debajo de cero, y entonces `_layer` cuenta una capa
        de más por caja: una caja puesta en la cubierta sale como capa 2, `measure` le
        busca apoyo en la capa 1 en vez de en el palé, y la traza dice 0 % de apoyo y
        margen negativo sobre un montón que está perfectamente plano. Medido.
        """
        if not getattr(measured, "placed", False):
            return
        # Ya viene sobre la cubierta: aquí NO se resta `deck_z`.
        self._levels.append(float(measured.position[2] - measured.spec.dims_m[2] / 2))
        # El CoG de la caja, no su centro: un bulto descentrado sigue pesando donde pesa.
        # Y de vuelta al mundo, que es el frame en el que `placing` compara con el centro
        # del palé.
        center_x, center_y = self._empty_state.pallet_center_xy
        cog = np.asarray(measured.spec.cog_offset_m, dtype=float)
        yaw = float(measured.yaw)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        world = (
            float(center_x + measured.position[0] + cog[0] * cosine - cog[1] * sine),
            float(center_y + measured.position[1] + cog[0] * sine + cog[1] * cosine),
        )
        mass = float(measured.spec.mass_kg)
        total = self.state.mass + mass
        if total <= 0.0:
            return
        self.state = PalletState(
            mass=total,
            com_xy=(
                (self.state.mass * self.state.com_xy[0] + mass * world[0]) / total,
                (self.state.mass * self.state.com_xy[1] + mass * world[1]) / total,
            ),
            pallet_center_xy=self._empty_state.pallet_center_xy,
            pallet_half_diag=self._empty_state.pallet_half_diag,
        )

    def set_forecast(self, specs: list[PackageSpec]) -> None:
        """`placing` no hace lookahead. Fingirlo sería mentir sobre lo que decide."""
        del specs

    # ── lo que `placing` no produce ──────────────────────────────────────────

    def _layer(self, z_base: float) -> int:
        """La capa, contando cuántos niveles de apoyo DISTINTOS quedan por debajo.

        No vale una altura de capa nominal: el catálogo va de 100 a 260 mm de alto y a
        la segunda caja el número deja de significar nada.

        Y no vale `set()`: dos cajas de la misma capa asientan a alturas que difieren en
        micras, así que como conjunto son dos valores distintos y la capa se dispara una
        por caja. Hay que agrupar con la MISMA tolerancia con la que se decide si dos
        alturas son la misma capa.
        """
        base = z_base - self.deck_z
        distinct, last = 0, None
        for level in sorted(lvl for lvl in self._levels if lvl < base - LAYER_TOL):
            if last is None or level - last > LAYER_TOL:
                distinct += 1
                last = level
        return 1 + distinct

    def _slot(self, result, layer: int) -> str:
        """El hueco, por su celda en la rejilla. Sale en el feed de la interfaz."""
        index = pick_scored_index(result.batch, result.scores,
                                  self.planner.scoring.tie_eps)
        if index is None:
            return f"score-L{layer}"
        return (f"score-L{layer}-{int(result.batch.i0[index])}"
                f"x{int(result.batch.j0[index])}")
