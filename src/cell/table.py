"""Fuente de mesa: los bultos esperan inmóviles hasta que el robot los recoge."""

from __future__ import annotations

import numpy as np

from src.cell.conveyor import _BaseSupply


class TableSupply(_BaseSupply):
    """Caso base: presentar un paquete no altera su pose sobre la mesa."""

    def stage(self, scene) -> None:
        cfg = scene.cfg["table"]
        center_x, center_y = (float(v) for v in cfg["center"])
        half_x, half_y, _ = (float(v) for v in cfg["size"])
        top = float(cfg["height"])
        jitter = scene.level.pos_jitter_m
        spread = np.radians(scene.level.yaw_jitter_deg)

        columns = max(1, int(len(scene.boxes) ** 0.5 + 0.5))
        step_x = (2 * half_x - 0.20) / max(1, columns - 1) if columns > 1 else 0.0
        rows = int(np.ceil(len(scene.boxes) / columns))
        step_y = (2 * half_y - 0.20) / max(1, rows - 1) if rows > 1 else 0.0

        for box in scene.boxes:
            column, row = box.index % columns, box.index // columns
            x = center_x - (2 * half_x - 0.20) / 2 + column * step_x
            y = center_y - (2 * half_y - 0.20) / 2 + row * step_y
            if jitter > 0:
                x += float(scene.rng.uniform(-jitter, jitter))
                y += float(scene.rng.uniform(-jitter, jitter))
            yaw = float(scene.rng.uniform(-spread, spread)) if spread > 0 else 0.0
            scene.place_box(box.index, (x, y, top + box.dims_m[2] / 2 + 0.002), yaw)
        scene.settle(0.4)

    def present(self, scene) -> str | None:
        """Devuelve el siguiente bulto sin moverlo; una mesa no puede atascarse."""
        if not self.pending:
            self.exhausted = True
            return None
        self.current = self.pending[0]
        return scene.boxes[self.current].package_id
