"""
El stub-planificador: rellena en rejilla y a correr.

No puntúa nada. Recorre una rejilla fija sobre el palé, coloca el paquete en el primer
hueco libre y pasa de capa cuando se acaban. Es deliberadamente tonto: existe para que
el bucle entero funcione mientras la heurística se escribe, y para tener **una línea
base contra la que comparar** cuando exista. Un score que no bate a rellenar en rejilla
no está aportando nada, y eso hay que poder demostrarlo.

Un run con este planificador NO es un run con planificación: va con `oracle=True` como
cualquier otro stub. Lo calcula `scripts/palletize.py`.

A implementar (veinte líneas):

    class GridPlanner:
        def __init__(self, pallet_cfg: dict): ...
        def choose(self, spec, heightmap) -> PlacementPlan | None:
            # siguiente celda libre de la rejilla; z = altura de la capa actual;
            # score fijo 0.0 y breakdown vacío, que es lo honesto: no puntúa.

Ojo con la holgura entre cajas: la rejilla tiene que respetarla o el episodio muere a la
tercera caja. En el eje de cierre de la pinza la impone el dedo, no la caja — el porqué
y los milímetros, en la cabecera de `configs/pallet.yaml`.
"""

from __future__ import annotations

from src.contracts import Heightmap, PackageSpec, PlacementPlan
