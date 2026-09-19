"""
La heurística: puntuar cada hueco candidato y quedarse con el mejor.

SIN IMPLEMENTAR. Los términos del score y sus pesos son la parte de diseño del módulo;
van en `configs/pallet.yaml` bajo `heuristic:`, no incrustados aquí, porque se van a
tocar mucho y cada cambio tiene que poder explicarse.

Entra un `PackageSpec` —el paquete que el brazo YA tiene agarrado y ya ha medido— y un
`Heightmap` recién medido. Sale un `PlacementPlan`, o `None`.

Cuatro cosas que van en el diseño desde el principio:

**El score devuelve su desglose.** `PlacementPlan.breakdown` lleva cada término por
separado, y viaja al `payload` del evento `plan` junto a `layer` y `slot` (en el payload
sobrar es inocuo). Es lo que permite responder "¿por qué puso esa caja ahí?" sin volver
a correr el episodio. Un score que solo devuelve un número obliga a depurar a ciegas.

**`None` no es una excepción.** Que no quepa en ningún sitio es un resultado normal del
planificador y se registra como `wrong_placement`; lanzar desde aquí tumba el episodio
entero y deja la fila en `running`.

**El CoG del paquete es un término, no un filtro.** Un bulto con el centro de gravedad
descentrado apoyado al borde del montón es un derrumbe con retraso: penalízalo y deja
que compita con el resto de términos. Descartarlo del todo hace que no quepa nada en
cuanto el palé va medio lleno.

**El CoG del palé también cuenta.** El objetivo no es rellenar huecos, es que el montón
llegue entero al camión: un hueco que deja el centro de gravedad del conjunto lejos del
centro del palé es peor que uno que desperdicia 2 cm. `stability_margin` de
`theker_telemetry` es el número con el que lo mide la interfaz — úsalo, no inventes otro
o tendrás dos verdades.

Términos de los que partir (y que hay que medir, no razonar):

    apoyo         fracción de la huella que queda sobre algo sólido
    planitud      diferencia de altura bajo la huella
    vuelo         cuánto sobresale del palé
    compacidad    hueco desperdiciado alrededor
    cog_paquete   cuánto descentra este paquete su propio apoyo
    cog_palet     a dónde deja el centro de gravedad del montón
"""

from __future__ import annotations

from src.contracts import Heightmap, PackageSpec, PlacementPlan


class ScorePlanner:
    """Heurística de score. Cumple `contracts.Planner`."""

    def __init__(self, pallet_cfg: dict):
        self.cfg = pallet_cfg

    def choose(self, spec: PackageSpec,
               heightmap: Heightmap) -> PlacementPlan | None:
        raise NotImplementedError("heurística sin implementar: usa planner.naive")
