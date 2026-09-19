"""
Medición: el paquete ya está en la mano, ¿cómo de grande es, cuánto pesa, dónde tiene
el centro de gravedad?

SIN IMPLEMENTAR. Cómo se mide está sin decidir —estación dedicada a la que el brazo
lleva el paquete, o medición en vuelo con los sensores del propio robot— y la decisión
se escribe aquí arriba cuando se tome. El contrato sí está cerrado: `measure()` devuelve
un `PackageSpec`.

Esto ocurre DESPUÉS de coger y ANTES de planificar, y ése es el orden que define la
arquitectura: el planificador no puede precalcular el palé entero porque no sabe cómo es
el siguiente paquete hasta que lo tiene agarrado.

**El CoG es la parte que justifica que este módulo exista.** Se estima —del par en la
muñeca, de dónde queda el paquete al colgar, de lo que sea que se decida—, no se lee del
catálogo. Un `cog_offset_m` copiado de `configs/pallet.yaml` es un oráculo con otro
nombre, y hace que la heurística parezca funcionar por un motivo que no es el suyo.

Lo que se mide de verdad y lo que se da por sabido tiene que quedar explícito aquí: la
masa, por ejemplo, puede salir del par de sujeción o puede darse por conocida del
albarán. Las dos son defendibles; lo que no lo es es no saber cuál se está haciendo.
"""

from __future__ import annotations

from src.contracts import Observation, PackageSpec


class WristGauge:
    """Medición con el paquete agarrado. Cumple `contracts.Gauge`."""

    def measure(self, scene, arm, observation: Observation) -> PackageSpec:
        raise NotImplementedError("medición sin implementar: usa vision.oracle")
