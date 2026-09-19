"""Fuente de mesa: los bultos esperan inmóviles hasta que el robot los recoge.

── POR QUÉ ESTO NO ES UNA REJILLA ───────────────────────────────────────────────

Lo era, y repartía la mesa entre el NÚMERO de cajas sin mirar cuánto medían. Con un
catálogo de un solo tipo colaba; en cuanto el nivel mezcla tamaños, no:

  - Nivel 12, seis tipos distintos: la separación en Y salía de 240 mm para cajas de
    hasta 340 mm de fondo. Se empujaban entre ellas y **seguían deslizándose cuando el
    episodio ya había empezado** —0.019 m/s medidos tras el asentado—, así que la caja ya
    no estaba donde se la vio: llegaba a la ventosa 23.8 mm corrida y el episodio moría
    en `wrong_placement` por 2 mm de tolerancia.
  - Nivel 13, ocho cajas sorteadas de hasta 550 mm: la mitad quedaba colgando del canto.
    Velocidad residual tras el asentado, **3.89 m/s**, que es caída libre.

El síntoma parecía del brazo —"se le escurre la caja"— y no lo era. Por eso aquí se
coloca por la HUELLA de cada bulto y se asienta hasta reposo de verdad, no durante un
tiempo fijo.

── Y POR QUÉ NO CABEN TODAS ─────────────────────────────────────────────────────

La mesa tiene 0.72 x 0.68 m útiles. Ocho bultos del generador suman más área que eso, así
que no es cuestión de colocarlos mejor: **no caben**. Los que no caben esperan aparcados
fuera de la escena y entran en el hueco que deja el anterior, que es lo que hace un
operario cuando la mesa está llena. La alternativa —fingir que caben— es la que los
mandaba al suelo.
"""

from __future__ import annotations

import numpy as np

from src.cell.conveyor import _BaseSupply

# Separación entre bultos vecinos en la mesa. No es la del palé: aquí no la impone la
# herramienta sino el asentado, y con menos de esto dos cajas con `yaw_jitter` se tocan
# por las esquinas y se empujan durante todo el episodio.
GAP = 0.03


class TableSupply(_BaseSupply):
    """Caso base: presentar un paquete no altera su pose sobre la mesa."""

    def stage(self, scene) -> None:
        cfg = scene.cfg["table"]
        center_x, center_y = (float(v) for v in cfg["center"])
        half_x, half_y, _ = (float(v) for v in cfg["size"])
        self.top = float(cfg["height"])
        self.spread = np.radians(scene.level.yaw_jitter_deg)
        self.jitter = scene.level.pos_jitter_m
        self.area = (center_x - half_x, center_y - half_y, 2 * half_x, 2 * half_y)
        self.cursor = (0.0, 0.0, 0.0)

        self.slots: dict[int, tuple[float, float]] = {}
        for box in scene.boxes:
            slot = self._next_slot(box)
            if slot is None:
                break                      # la mesa se llenó; el resto espera aparcado
            self.slots[box.index] = slot
            self._put(scene, box, slot)
        scene.settle_until_rest()

    def present(self, scene) -> str | None:
        """Devuelve el siguiente bulto sin moverlo; una mesa no puede atascarse."""
        if not self.pending:
            self.exhausted = True
            return None
        self.current = self.pending[0]
        box = scene.boxes[self.current]
        if self.current not in self.slots:
            # No cabía cuando se preparó la mesa. Entra ahora, en el hueco que dejó el
            # anterior, y se le deja asentar antes de que nadie lo mire.
            slot = self._recycle_slot(box)
            if slot is None:
                return None                # sin hueco: la espera expira como `timeout`
            self.slots[self.current] = slot
            self._put(scene, box, slot)
            scene.settle_until_rest()
        return box.package_id

    # ── colocar sin pisarse ──────────────────────────────────────────────────

    def _footprint(self, box) -> tuple[float, float]:
        """La huella con el giro contado: `yaw_jitter` la ensancha, y hay que pagarlo."""
        length, width, _ = box.dims_m
        cosine, sine = abs(np.cos(self.spread)), abs(np.sin(self.spread))
        span_x = length * cosine + width * sine
        span_y = length * sine + width * cosine
        pad = GAP + 2 * self.jitter
        return float(span_x) + pad, float(span_y) + pad

    def _next_slot(self, box) -> tuple[float, float] | None:
        """Empaquetado por estantes: se llena una fila y se baja a la siguiente."""
        span_x, span_y = self._footprint(box)
        origin_x, origin_y, width, depth = self.area
        cursor_x, cursor_y, row_depth = self.cursor
        if cursor_x + span_x > width:                      # no cabe en esta fila
            cursor_x, cursor_y, row_depth = 0.0, cursor_y + row_depth, 0.0
        if cursor_x + span_x > width or cursor_y + span_y > depth:
            return None                                    # ni en la mesa
        slot = (origin_x + cursor_x + span_x / 2, origin_y + cursor_y + span_y / 2)
        self.cursor = (cursor_x + span_x, cursor_y, max(row_depth, span_y))
        return slot

    def _recycle_slot(self, box) -> tuple[float, float] | None:
        """Un hueco libre para esta caja: el de la anterior, o la mesa entera si está vacía.

        Exigir el hueco EXACTO de la que se fue no vale con bultos sorteados: el
        siguiente casi nunca es del mismo tamaño y la mesa se quedaba sin poder servir
        nada, que salía como `timeout` cuando lo que pasaba es que sobraba sitio.
        """
        span_x, span_y = self._footprint(box)
        origin_x, origin_y, width, depth = self.area
        half_x, half_y = span_x / 2, span_y / 2
        for index, slot in list(self.slots.items()):
            if index in self.pending and index != self.current:
                continue                                   # sigue ocupado
            if (origin_x <= slot[0] - half_x and slot[0] + half_x <= origin_x + width
                    and origin_y <= slot[1] - half_y
                    and slot[1] + half_y <= origin_y + depth):
                del self.slots[index]
                return slot
        # Nada reutilizable. Si no queda ningún bulto puesto, la mesa está libre entera y
        # se reparte de cero; si queda alguno, no hay sitio y la espera expirará.
        if any(index in self.pending and index != self.current for index in self.slots):
            return None
        self.slots.clear()
        self.cursor = (0.0, 0.0, 0.0)
        return self._next_slot(box)

    def _put(self, scene, box, slot: tuple[float, float]) -> None:
        x, y = slot
        if self.jitter > 0:
            x += float(scene.rng.uniform(-self.jitter, self.jitter))
            y += float(scene.rng.uniform(-self.jitter, self.jitter))
        yaw = float(scene.rng.uniform(-self.spread, self.spread)) if self.spread > 0 else 0.0
        scene.place_box(box.index, (x, y, self.top + box.dims_m[2] / 2 + 0.002), yaw)
