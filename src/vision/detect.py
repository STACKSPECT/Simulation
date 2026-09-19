"""
Percepción: qué hay parado en la estación de recogida de la cinta.

SIN IMPLEMENTAR. El enfoque está sin decidir (RGB-D renderizada y segmentación clásica,
buffer de segmentación degradado con ruido, o un modelo entrenado) y esa decisión no la
toma este fichero: la toma el equipo y se escribe aquí arriba cuando se tome.

Lo que NO está sin decidir es el contrato: `observe()` devuelve `list[Observation]` y
nada más. Mientras no exista, `src/vision/oracle.py` cubre el hueco.

Dos cosas que este módulo tiene que poder hacer, y que el oráculo no hace:

  - **Equivocarse.** Devolver la lista vacía es `no_detection`, que en el repo
    guionizado era inalcanzable y aquí es un resultado legítimo. Si nunca puede pasar,
    no es percepción.
  - **Dudar.** `confidence` tiene que significar algo. Va al payload del evento
    `perceive`, y un 1.0 constante es la firma de un oráculo disfrazado.

El renderer sí conviene mantenerlo vivo entre pasadas, al revés que en `cell/render.py`:
aquí se renderiza en cada paquete, no dos veces por episodio.
"""

from __future__ import annotations

from src.contracts import Observation


class CameraDetector:
    """Detector sobre la cámara de la celda. Cumple `contracts.Detector`."""

    def observe(self, scene) -> list[Observation]:
        raise NotImplementedError("percepción sin implementar: usa vision.oracle")
