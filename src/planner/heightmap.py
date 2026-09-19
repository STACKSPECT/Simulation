"""
El mapa de alturas del palé: cuánto mide el montón en cada celda.

SIN IMPLEMENTAR.

**Se mide del estado de la escena, no se lleva en un contador.** Si el planificador
arrastra su propia idea de cómo está el palé —"aquí puse una caja de 45 mm, luego aquí
hay 45 mm"—, a la tercera caja torcida deja de coincidir con la realidad y ya no se
recupera: sigue planificando sobre un palé imaginario mientras el de verdad se derrumba.
Medirlo cuesta una pasada sobre las cajas que están sobre el palé y vale lo que cuesta.

Es el único fichero de `src/planner/` que puede mirar el simulador. El resto recibe el
`Heightmap` ya hecho, y eso es lo que permite probar la heurística sin arrancar MuJoCo.

Dos decisiones que hay que tomar y dejar escritas aquí:

  - **El tamaño de celda.** Fino de más y cada caja torcida mete ruido de dientes de
    sierra; grueso de más y un hueco de 20 mm parece plano. Empieza por algo del orden
    del lado menor del paquete más pequeño y mídelo, no lo estimes.
  - **Qué hacer con las cajas a medio caer.** Una caja inclinada no tiene una altura,
    tiene un rango. Quedarse con el máximo es pesimista y es lo que conviene: el brazo
    va a chocar con el punto alto, no con la media.
"""

from __future__ import annotations

from src.contracts import Heightmap


def measure(scene) -> Heightmap:
    """Altura del montón sobre la cubierta, celda a celda."""
    raise NotImplementedError("mapa de alturas sin implementar")
