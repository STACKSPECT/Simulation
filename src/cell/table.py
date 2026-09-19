"""Fuente de mesa: un bulto preparado, y el siguiente entra cuando se lo llevan.

── POR QUÉ UNO, Y NO LA MESA LLENA ──────────────────────────────────────────────

Estaba llena, repartida en rejilla entre el NÚMERO de bultos y sin mirar cuánto medían.
Eso daba tres cosas, y ninguna se veía como lo que era:

  - **Bultos solapados.** Nivel 12, seis tipos: 240 mm de separación para cajas de 340 mm
    de fondo. Seguían deslizándose con el episodio ya empezado —0.019 m/s tras el
    asentado— así que la caja llegaba a la ventosa 23.8 mm corrida y el episodio moría en
    `wrong_placement` por 2 mm de tolerancia. Parecía que se escurría de la ventosa.
  - **Bultos colgando del canto.** Nivel 13, ocho sorteados de hasta 550 mm: velocidad
    residual 3.89 m/s, que es caída libre.
  - **Y el vecino por los aires.** Éste sólo sale con movimiento real; en fast-forward el
    brazo teletransporta y no se ve. Al extraer un bulto, el que lleva en la mano barre
    al de al lado y lo tira al suelo.

Esto último es lo que decide el diseño, porque **no tiene arreglo con más holgura**. La
mesa mide 0.72 x 0.68 m útiles. Dos `std_m` (0.42 x 0.30) sólo entran de canto en Y, y
ahí la holgura máxima que cabe son 80 mm:

    holgura   vecina tras extraer      cabe en la mesa
     30 mm    384 mm, al suelo         sí
     80 mm    385 mm, al suelo         sí (al límite)
    150 mm    —                        NO, se sale del canto

Es decir: la holgura que hace falta para extraer sin tocar es mayor que la que cabe. Con
esta mesa y esta herramienta, la única configuración que no derriba nada es **un bulto**.
El resto espera aparcado fuera de la escena y entra cuando el anterior se va, que es lo
que hace un operario con una mesa pequeña.

Si se quiere la mesa llena, lo que hay que cambiar es la mesa —o sacar el bulto por
arriba antes de trasladar—, no el número de la holgura.
"""

from __future__ import annotations

import numpy as np

from src.cell.conveyor import _BaseSupply


class TableSupply(_BaseSupply):
    """Un bulto preparado en el centro de la mesa. Presentarlo no lo mueve."""

    def stage(self, scene) -> None:
        cfg = scene.cfg["table"]
        self.center = tuple(float(v) for v in cfg["center"])
        half_x, half_y, _ = (float(v) for v in cfg["size"])
        self.half = (half_x, half_y)
        self.top = float(cfg["height"])
        self.spread = np.radians(scene.level.yaw_jitter_deg)
        self.jitter = scene.level.pos_jitter_m
        self.staged: int | None = None
        if scene.boxes:
            self._put(scene, scene.boxes[0])
        scene.settle_until_rest()

    def present(self, scene) -> str | None:
        """Devuelve el bulto de la mesa; una mesa no puede atascarse."""
        if not self.pending:
            self.exhausted = True
            return None
        self.current = self.pending[0]
        if self.staged != self.current:
            self._put(scene, scene.boxes[self.current])
            scene.settle_until_rest()
        return scene.boxes[self.current].package_id

    def _put(self, scene, box) -> None:
        """Al centro de la mesa, que es donde más canto le queda por los cuatro lados."""
        x, y = self.center
        if self.jitter > 0:
            x += float(scene.rng.uniform(-self.jitter, self.jitter))
            y += float(scene.rng.uniform(-self.jitter, self.jitter))
        yaw = float(scene.rng.uniform(-self.spread, self.spread)) if self.spread else 0.0
        scene.place_box(box.index, (x, y, self.top + box.dims_m[2] / 2 + 0.002), yaw)
        self.staged = box.index

    def fits(self, box) -> bool:
        """Si el bulto cabe entero en la mesa. Uno que sobresale se cae y se mide en el suelo."""
        cosine, sine = abs(np.cos(self.spread)), abs(np.sin(self.spread))
        span_x = box.dims_m[0] * cosine + box.dims_m[1] * sine
        span_y = box.dims_m[0] * sine + box.dims_m[1] * cosine
        return (span_x / 2 + self.jitter <= self.half[0]
                and span_y / 2 + self.jitter <= self.half[1])
