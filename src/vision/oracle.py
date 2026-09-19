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

from src.contracts import Observation, PackageSpec

# A implementar cuando exista `cell/scene.py` (portado) y `cell/conveyor.py`:
#
#   class OracleDetector:
#       """Devuelve la caja que la cinta tiene parada en la estación, sin mirar imagen
#       ninguna. `confidence=1.0` y `dims_guess` = las dimensiones reales."""
#       def observe(self, scene) -> list[Observation]: ...
#
#   class OracleGauge:
#       """Lee `dims`, `mass` y `cog_offset` del catálogo de la escena. No mide nada."""
#       def measure(self, scene, arm, observation) -> PackageSpec: ...
#
# Son diez líneas cada uno. Lo que no puede pasar es que se queden y se olviden: en
# cuanto exista la implementación buena, el stub sigue aquí para poder comparar, pero
# deja de ser el de por defecto en `scripts/palletize.py`.
