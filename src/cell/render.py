"""
Las vistas del palé que se suben a la plataforma.

El renderer se crea y se destruye en cada llamada: esto pasa unas pocas veces por
episodio y no compensa mantener vivo un contexto GL entre medias. En percepción es al
revés —se renderiza en cada paquete— y por eso vive en otro sitio.

── `view` ES UN VOCABULARIO CERRADO ─────────────────────────────────────────────

`top | side | iso | camera`, y nada más. Con otro nombre el PNG sube a Storage y
**luego** la base rechaza la fila con un `23514`: la foto queda huérfana en el bucket y
la traza se queda sin imagen. Al repo anterior le pasó exactamente con `front`.

Por eso la comprobación está en dos sitios: aquí, y en `tests/test_pallet.py`, que la
hace sin arrancar el simulador. Las cámaras de `configs/pallet.yaml` tienen que llamarse
con uno de esos cuatro nombres o la escena ni siquiera compila.

── Y LO DE SIEMPRE CON LAS FOTOS ────────────────────────────────────────────────

**El brazo se aparta primero, y en CARTESIANO.** Acaba justo encima del palé, que es
donde estaba soltando, así que sin apartarlo la cenital sale del dorso de la mano. Y
apartarlo en espacio de juntas deja el recorrido sin controlar y barre el montón recién
colocado: medido en el repo anterior, el episodio pasaba de 10/10 a 4/10 con
`overhang_violation` en cuanto se metió la foto por capa.

La maniobra vive en `src/episode.py`; aquí sólo se dispara el obturador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Vocabulario cerrado de `snapshots.view`. Ampliarlo obliga a tocar el CHECK de
# `backend/sql/003_snapshots.sql` en Platform: pídeselo a quien lleve el backend.
VIEWS = ("top", "side", "iso", "camera")


@dataclass(frozen=True)
class Snapshot:
    """Una foto tomada, lista para subir."""

    view: str
    after_seq: int
    image: np.ndarray


# Cuántos prismas como mucho se pintan para el mapa de alturas. La rejilla real es de
# 120x80 = 9600 celdas y el `user_scn` del visor no da para tanto, así que se submuestrea
# hasta caber. Es una ayuda para mirar, no una medida: lo medido va en la telemetría.
HEIGHTMAP_BUDGET = 900


def draw_heightmap(scene, heightmap) -> None:
    """Pinta el mapa de alturas en el visor. La tecla `m` lo enciende y lo apaga.

    Un prisma por celda, de la altura que la celda dice. Verde abajo, rojo arriba, y las
    celdas que ninguna cámara vio en azul translúcido: ésas guardan altura 0 y NO son
    cubierta libre, que es justo lo que no se ve de ninguna otra forma.

    No hace nada sin visor, así que es seguro llamarlo siempre.
    """
    viewer = getattr(scene, "viewer", None)
    if viewer is None:
        return
    mujoco = scene.mujoco
    scn = viewer.user_scn
    scn.ngeom = 0
    if not getattr(scene, "show_heightmap", False):
        viewer.sync()
        return

    cells = heightmap.cells
    observed = heightmap.observed
    ny, nx = cells.shape
    cell = float(heightmap.cell_size)
    origin_x, origin_y = heightmap.origin
    budget = min(HEIGHTMAP_BUDGET, int(scn.maxgeom))
    step = 1
    while (nx // step) * (ny // step) > budget:
        step += 1
    side = cell * step
    ceiling = max(float(scene.cfg["heuristic"]["max_stack_height"]), 1e-6)

    identity = np.eye(3, dtype=np.float64).reshape(9)
    count = 0
    for j in range(0, ny - step + 1, step):
        for i in range(0, nx - step + 1, step):
            if count >= budget:
                break
            patch = cells[j:j + step, i:i + step]
            height = float(patch.max())
            seen = True if observed is None else bool(observed[j:j + step, i:i + step].any())
            # Una celda sin observar se dibuja como una lámina fina sobre la cubierta: lo
            # que importa de ella no es su altura, es que no se sabe.
            thickness = max(height, 0.004) if seen else 0.004
            ratio = min(height / ceiling, 1.0)
            rgba = (ratio, 1.0 - ratio, 0.15, 0.40) if seen else (0.15, 0.35, 1.0, 0.55)
            mujoco.mjv_initGeom(
                scn.geoms[count],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                # El 0.92 deja una junta entre celdas vecinas. Sin él las caras laterales
                # de dos celdas contiguas caen en el mismo plano, el z-buffer no sabe
                # cuál va delante y la malla sale moteada de puntos claros que parpadean
                # al girar la cámara. Con la junta se ve la rejilla y no parpadea nada.
                size=np.array([side * 0.46, side * 0.46, thickness / 2]),
                pos=np.array([
                    origin_x + (i + step / 2) * cell,
                    origin_y + (j + step / 2) * cell,
                    scene.deck_z + thickness / 2,
                ]),
                mat=identity,
                rgba=np.array(rgba, dtype=np.float32),
            )
            count += 1
    scn.ngeom = count
    viewer.sync()


def render(scene, view: str) -> np.ndarray:
    """Un fotograma RGB desde la cámara `view`, con el palé ya asentado.

    Devuelve el array; convertir a PNG es cosa de `src/telemetry.py`, que es el único
    módulo que sabe que la plataforma existe.
    """
    if view not in VIEWS:
        raise ValueError(
            f"vista {view!r} fuera del vocabulario {VIEWS}. Con otro nombre el PNG sube "
            f"a Storage y luego la base rechaza la fila con un 23514."
        )
    spec = scene.cfg["cameras"][view]
    width, height = spec["resolution"]

    renderer = scene.mujoco.Renderer(scene.model, height=height, width=width)
    try:
        renderer.update_scene(scene.data, camera=view)
        return renderer.render()
    finally:
        # Sin esto el contexto GL sobrevive a la llamada y, tras unas cuantas capas, el
        # proceso se queda sin contextos. Son pocas fotos: abrir y cerrar sale barato.
        renderer.close()
