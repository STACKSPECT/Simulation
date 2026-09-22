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

from src.cell.scene import PACKAGE_GROUP

# Vocabulario cerrado de `snapshots.view`. Ampliarlo obliga a tocar el CHECK de
# `backend/sql/003_snapshots.sql` en Platform: pídeselo a quien lleve el backend.
VIEWS = ("top", "side", "iso", "camera")

# Los centros de masa del visor, con los tamaños del demostrador
# (`tools/stable_pallet/com_markers.py`). Los de cada bulto llevan los colores de las
# casillas del panel —verde lo real, naranja lo calculado—; los de la pila, otros dos,
# porque sin texto encima el color es lo único que los distingue. El amarillo es el
# error entre las dos pilas.
TRUE_COM_RGBA = (0.16, 0.92, 0.45, 0.95)
ESTIMATED_COM_RGBA = (1.00, 0.46, 0.10, 0.95)
TRUE_LOAD_RGBA = (0.20, 0.55, 1.00, 0.95)
ESTIMATED_LOAD_RGBA = (0.93, 0.28, 0.85, 0.95)
ERROR_RGBA = (0.98, 0.86, 0.22, 0.85)
GHOST_ALPHA = 0.25
PACKAGE_RADIUS = 0.018
LOAD_RADIUS = 0.034
PLUMB_WIDTH = 0.004
ERROR_WIDTH = 0.006


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


def silence_com_markers(viewer, mujoco) -> None:
    """Apaga las esferas blancas del centro de masas del visor.

    **`m` ya era un atajo de MuJoCo**: el visor tiene una letra asignada a cada bandera
    de visualización y `M` es "Center of Mass", que pinta una esfera blanca en el CoM de
    cada cuerpo. Las veintiséis letras están cogidas, así que la rejilla no tiene ninguna
    libre donde meterse; lo que se hace es quedarse con la tecla y dejar la bandera de
    MuJoCo apagada, en vez de compartirla y que las dos se desincronicen —que es
    exactamente lo que pasaba: una pulsación encendía las esferas y apagaba la rejilla—.

    Se llama en cada sincronización, no sólo al pulsar, porque el visor procesa su propia
    tecla al margen de este callback y no hay forma de saber quién va primero.
    """
    opt = getattr(viewer, "opt", None)
    if opt is None:
        return
    opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0


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
    silence_com_markers(viewer, mujoco)
    scn = viewer.user_scn
    scn.ngeom = 0
    scene.heightmap_geoms = 0
    if not getattr(scene, "show_heightmap", False):
        draw_overlay(scene)
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
    scene.heightmap_geoms = count
    draw_overlay(scene)
    viewer.sync()


def draws_com(scene) -> bool:
    """Si alguna de las dos casillas de centro de masa está encendida."""
    return bool(getattr(scene, "show_true_com", False)
                or getattr(scene, "show_estimated_com", False))


def draw_overlay(scene) -> None:
    """Pinta los centros de masa, y los bultos en fantasma para verlos.

    **Verde**, dónde está de verdad el peso de cada bulto (`xipos`, lo que integra
    MuJoCo), y en todos desde el principio: esperando, en la fuente o en la mano.
    **Naranja**, dónde cree la celda que está: la pose del bulto más el `cog_offset_m`
    que midió el gauge. Sale al pesarlo, con el bulto ya en la mano y antes de
    planificar, porque antes no hay medida que pintar; la distancia al verde es el error
    con el que el planificador elige hueco.

    Las esferas grandes, con su plomada hasta la cubierta, son el CoG de la PILA:
    **azul** el real y **magenta** el calculado, y la línea **amarilla** entre los dos es
    el error. Sólo cuentan los bultos que `src/episode.py` da por puestos en el palé,
    los mismos con los que `measure.pallet_state` saca la traza de CoG.

    **Un CoG está DENTRO de su cartón**, así que con los bultos opacos las esferas no se
    ven. Al abrir el visor con marcadores, `palletize.py` apaga el grupo de los bultos
    (`PACKAGE_GROUP`) y aquí se pintan en su lugar unas copias translúcidas. No se toca
    el `rgba` del modelo: lo leen también las fotos y las cámaras de profundidad, y un
    bulto translúcido sale distinto en la foto y puede desaparecer del mapa medido. Con
    el grupo encendido —la tecla `1`— vuelven los bultos opacos y las copias sobran.

    Va detrás de la rejilla del mapa de alturas en el mismo `user_scn`: la rejilla sólo
    cambia al medir y ocupa las primeras `scene.heightmap_geoms` casillas, y esto se
    repinta en cada fotograma porque los bultos se mueven. Sin marcadores no toca nada.
    """
    viewer = getattr(scene, "viewer", None)
    if viewer is None or not draws_com(scene):
        return
    mujoco = scene.mujoco
    model, data = scene.model, scene.data
    scn = viewer.user_scn
    scn.ngeom = int(getattr(scene, "heightmap_geoms", 0))
    identity = np.eye(3, dtype=np.float64).reshape(9)

    def next_geom():
        if scn.ngeom >= scn.maxgeom:
            return None
        geom = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        return geom

    def sphere(position, radius: float, rgba) -> None:
        geom = next_geom()
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.asarray(position, dtype=np.float64),
            mat=identity,
            rgba=np.array(rgba, dtype=np.float32),
        )

    def line(start, end, width: float, rgba) -> None:
        geom = next_geom()
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=identity,
            rgba=np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64),
        )

    opt = getattr(viewer, "opt", None)
    if opt is not None and not opt.geomgroup[PACKAGE_GROUP]:
        for box in scene.boxes:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, box.geom)
            geom = next_geom() if index >= 0 else None
            if geom is None:
                continue
            rgba = np.array(model.geom_rgba[index], dtype=np.float32)
            rgba[3] = min(float(rgba[3]), GHOST_ALPHA)
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array(model.geom_size[index], dtype=np.float64),
                pos=np.array(data.geom_xpos[index], dtype=np.float64),
                mat=np.array(data.geom_xmat[index], dtype=np.float64),
                rgba=rgba,
            )

    show_true = bool(getattr(scene, "show_true_com", False))
    show_estimated = bool(getattr(scene, "show_estimated_com", False))

    def true_com(index: int) -> np.ndarray:
        return np.array(data.xipos[scene.body_id(index)], dtype=float)

    def believed_com(index: int, spec) -> np.ndarray:
        body = scene.body_id(index)
        rotation = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
        offset = np.asarray(spec.cog_offset_m, dtype=float)
        return np.asarray(data.xpos[body], dtype=float) + rotation @ offset

    if show_true:
        for box in scene.boxes:
            sphere(true_com(box.index), PACKAGE_RADIUS, TRUE_COM_RGBA)
    if show_estimated:
        for index, spec in getattr(scene, "weighed_specs", {}).items():
            sphere(believed_com(index, spec), PACKAGE_RADIUS, ESTIMATED_COM_RGBA)

    load_placements = getattr(scene, "load_placements", [])
    true_points = [true_com(p.box.index) for p in load_placements]
    true_masses = [float(model.body_mass[scene.body_id(p.box.index)]) for p in load_placements]
    believed_points = [believed_com(p.box.index, p.spec) for p in load_placements]
    believed_masses = [float(p.spec.mass_kg) for p in load_placements]

    # La plomada baja a la cubierta del palé, no al suelo, y sigue su inclinación:
    # en el ensayo de estabilidad el palé se ladea y la plomada tiene que ladearse con él.
    pallet = np.asarray(data.xmat[scene.pallet_body], dtype=float).reshape(3, 3)
    deck = np.asarray(data.xpos[scene.pallet_body], dtype=float) + pallet[:, 2] * scene.deck_z
    normal = pallet[:, 2]

    def load(points, masses, rgba):
        total = sum(masses)
        if total <= 0.0:
            return None
        centre = sum(mass * point for mass, point in zip(masses, points)) / total
        foot = centre - normal * float(np.dot(centre - deck, normal))
        sphere(centre, LOAD_RADIUS, rgba)
        line(foot, centre, PLUMB_WIDTH, rgba)
        sphere(foot, PACKAGE_RADIUS * 0.8, rgba)
        return centre

    truth = load(true_points, true_masses, TRUE_LOAD_RGBA) if show_true else None
    belief = (
        load(believed_points, believed_masses, ESTIMATED_LOAD_RGBA)
        if show_estimated else None
    )
    if truth is not None and belief is not None:
        line(belief, truth, ERROR_WIDTH, ERROR_RGBA)


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
