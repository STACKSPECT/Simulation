"""
El mapa de alturas del palé: cuánto mide el montón en cada celda.

**Se mide del estado de la escena, no se lleva en un contador.** Si el planificador
arrastra su propia idea de cómo está el palé —"aquí puse una caja de 45 mm, luego aquí
hay 45 mm"—, a la tercera caja torcida deja de coincidir con la realidad y ya no se
recupera: sigue planificando sobre un palé imaginario mientras el de verdad se derrumba.

Es el único fichero de `src/planner/` que puede mirar el simulador. El resto recibe el
`Heightmap` ya hecho, y eso es lo que permite probar la heurística sin arrancar MuJoCo.

── DOS MEDIDAS, Y NO SON LA MISMA ───────────────────────────────────────────────

`measure()` levanta el mapa de las **cámaras**: tres vistas de profundidad, se
desproyectan, se rasterizan sobre la huella del palé y se fusionan quedándose con lo más
alto. Trae `observed`, que es lo que aporta medir de verdad: una celda tapada por el
montón guarda altura 0 y **no es cubierta libre**.

`measure_ground_truth()` lo lee de las poses de MuJoCo. Es el oráculo, y viene de
`planner/naive.py` tal cual estaba. Un run que lo use NO es un run con percepción: lo
elige `--oracle-vision` en `scripts/palletize.py`, que es quien calcula `oracle`.

── LAS DOS DECISIONES QUE HABÍA QUE DEJAR ESCRITAS ──────────────────────────────

  - **El tamaño de celda** es `heuristic.cell_size`, hoy 10 mm. **Sin barrer**: es el
    valor por defecto del módulo `placing`, no una medida de esta celda. Fino de más y
    cada caja torcida mete ruido de dientes de sierra; grueso de más y un hueco de 20 mm
    parece plano.
  - **Las cajas a medio caer** no tienen una altura, tienen un rango. Las dos medidas se
    quedan con el **máximo**, que es lo pesimista y lo que conviene: el brazo va a chocar
    con el punto alto, no con la media. En la de cámaras sale gratis (la fusión es un
    máximo por celda); en el oráculo es la envolvente alineada de la caja girada.
"""

from __future__ import annotations

import math

import numpy as np

from src.contracts import Heightmap
from src.vision import depth as depth_camera
from src.vision.surface import fuse_max, rasterize_points, unproject_depth


def _grid(scene) -> tuple[float, tuple[float, float], float, float, int, int]:
    """Celda, origen y huella del palé. Una sola definición para las dos medidas."""
    cell = float(scene.cfg["heuristic"]["cell_size"])
    length, width = scene.pallet_dims                 # X, Y, COMPLETAS
    center_x, center_y = scene.pallet_center
    origin = (center_x - length / 2, center_y - width / 2)
    return cell, origin, length, width, math.ceil(length / cell), math.ceil(width / cell)


def _camera_resolution(scene, name: str) -> tuple[int, int]:
    """La resolución de una cámara, esté en `cameras:` o en `perception.rig:`."""
    rig = scene.cfg["perception"].get("rig", {})
    spec = rig.get(name) or scene.cfg["cameras"][name]
    width, height = spec["resolution"]
    return int(width), int(height)


def measure(scene) -> Heightmap:
    """El montón visto por las cámaras de percepción, en metros sobre la cubierta."""
    cell, origin, length, width, _, _ = _grid(scene)
    perception = scene.cfg["perception"]

    maps = []
    for name in perception["cameras"]:
        frame = depth_camera.frame(scene, name, *_camera_resolution(scene, name))
        points = unproject_depth(
            frame.depth,
            frame.intrinsics,
            frame.pose,
            depth_min=float(perception["depth_min"]),
            depth_trunc=float(perception["depth_trunc"]),
            min_upward_nz=float(perception["min_upward_nz"]),
        )
        maps.append(
            rasterize_points(
                points,
                origin_xy=origin,
                length=length,
                width=width,
                resolution=cell,
                z_min=float(perception["z_min"]),
                z_max=float(perception["z_max"]),
            )
        )

    fused = fuse_max(maps)
    # Las cámaras dan Z del mundo; el contrato es metros SOBRE la cubierta. La cubierta
    # vista queda en 0, y lo que caiga por debajo —el suelo de alrededor colado por un
    # borde, ruido de rasterizado— se recorta ahí: negativo no significa nada apilando.
    cells = np.maximum(0.0, fused.heights.astype(float) - scene.deck_z)
    return Heightmap(
        cells=cells, origin=origin, cell_size=cell, observed=fused.observed
    )


def _stamp_static_obstacles(scene, cells, origin, cell, nx, ny) -> None:
    """Los muebles de la celda que se meten sobre la huella del palé.

    La mesa de recogida ocupa x∈[-0.56, 0.16] y el palé empieza en x=0.00: hay un
    rectángulo de la cubierta que NO se puede usar, y hasta ahora nadie lo decía. Un
    planificador que puntúa huecos se va derecho a esa esquina —está baja, está pegada al
    canto y apoya al 100%—, el brazo empuja la caja contra la mesa y el episodio muere en
    `ik_unreachable` sin que se entienda por qué. Medido: la caja se queda a 135 mm del
    destino con la mesa penetrada 39 mm.

    Las cámaras ven la mesa y la meten en el mapa ellas solas. Esto es para que el
    oráculo cuente la MISMA verdad: si las dos medidas discrepan, la comparación entre
    correr con percepción y correr sin ella deja de significar nada.

    La cinta y el remolque no hacen falta aquí: ninguno se solapa con la huella del palé.
    El remolque acaba en y=-0.40, justo en el borde. La cinta ya no se libra por la X
    —se alargó hasta x=+0.15 para que el cartón no pare volando sobre el canto— sino por
    la Y: ocupa y∈[-1.05, -0.55] y deja 150 mm hasta la cubierta. Ése es el motivo de que
    bajase a y=-0.80 al alargarla; si alguien la sube, esto deja de ser cierto.
    """
    table = scene.cfg.get("table")
    if not table:
        return
    center_x, center_y = (float(v) for v in table["center"])
    half_x, half_y, _ = (float(v) for v in table["size"])     # SEMIEJES
    top = float(table["height"]) - scene.deck_z
    if top <= 0.0:
        return
    x0 = max(0, math.floor((center_x - half_x - origin[0]) / cell))
    x1 = min(nx, math.ceil((center_x + half_x - origin[0]) / cell))
    y0 = max(0, math.floor((center_y - half_y - origin[1]) / cell))
    y1 = min(ny, math.ceil((center_y + half_y - origin[1]) / cell))
    if x0 >= x1 or y0 >= y1:
        return
    cells[y0:y1, x0:x1] = np.maximum(cells[y0:y1, x0:x1], top)


def measure_ground_truth(scene) -> Heightmap:
    """El oráculo: el montón leído de las poses de MuJoCo.

    Portado de `planner/naive.measured_heightmap` sin tocar las cuentas. Devuelve
    `observed=None`, que es lo honesto: aquí no se ha observado nada, se ha mirado la
    respuesta.
    """
    cell, origin, _, _, nx, ny = _grid(scene)
    cells = np.zeros((ny, nx), dtype=float)
    _stamp_static_obstacles(scene, cells, origin, cell, nx, ny)
    measured = []
    for box in scene.boxes:
        position, _ = scene.box_pose(box.index)
        if position[2] < scene.deck_z:
            continue
        measured.append((float(position[2] - box.dims_m[2] / 2), box, position))
    for bottom, box, position in sorted(measured, key=lambda item: item[0]):
        yaw = scene.box_yaw(box.index)
        cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
        span_x = box.dims_m[0] * cosine + box.dims_m[1] * sine
        span_y = box.dims_m[0] * sine + box.dims_m[1] * cosine
        x0 = max(0, math.floor((position[0] - span_x / 2 - origin[0]) / cell))
        x1 = min(nx, math.ceil((position[0] + span_x / 2 - origin[0]) / cell))
        y0 = max(0, math.floor((position[1] - span_y / 2 - origin[1]) / cell))
        y1 = min(ny, math.ceil((position[1] + span_y / 2 - origin[1]) / cell))
        if x0 >= x1 or y0 >= y1:
            continue
        # Una caja de la mesa o en la mano puede proyectarse sobre el borde del palé,
        # pero no forma parte del montón si no apoya en la altura ya medida.
        support = cells[y0:y1, x0:x1]
        if bottom > scene.deck_z + float(support.max(initial=0.0)) + 0.02:
            continue
        top = max(0.0, float(position[2] + box.dims_m[2] / 2 - scene.deck_z))
        cells[y0:y1, x0:x1] = np.maximum(cells[y0:y1, x0:x1], top)
    return Heightmap(cells=cells, origin=origin, cell_size=cell)
