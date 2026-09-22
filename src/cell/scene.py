"""
La escena: brazo, mesas, palé, la fuente de suministro, paquetes, cámaras y luces.

Portado desde `tools/stable_pallet/simulator.py`, que es el demostrador del que sale
toda la capa de ejecución de este repo.

    PalletScene, Box, Level
    load_configs()        los dos YAML, con `pallet.yaml` pisando a `scene.yaml`
    build_scene()         el modelo compilado y asentado
    box_pose(scene, i)    dónde está de verdad una caja

── POR QUÉ MJCF POR CADENA Y NO `MjSpec` ────────────────────────────────────────

La cabecera original decía de construir con `mujoco.MjSpec` sobre el `scene.xml` del
Menagerie, para no pelearse con `meshdir` ni con rutas de `<include>`. Aquí se hace con
MJCF generado, que `MIGRACION.md` §2 acepta explícitamente, por dos razones concretas:

1. El `scene.xml` del Menagerie trae su propio suelo, su propia luz y su propia cámara.
   Esta celda es una nave con palé, valla de seguridad, marcas de suelo y un remolque
   arrimado al muelle; incluirlo obligaba a deshacer casi todo lo que trae.
2. La celda se **reconstruye** cada vez que la herramienta sella un cartón: el paquete
   deja de ser un cuerpo libre y pasa a colgar de `tool`. Eso cambia el árbol de
   cuerpos, no sólo un parámetro. Con una plantilla de texto es volver a compilar; con
   un `MjSpec` vivo es mutar el árbol con el visor mirando.

Del Menagerie se reutiliza lo que importa: las veinte mallas OBJ del UR10e y sus cifras
de cinemática, masas e inercias. La ruta está en `configs/scene.yaml: robot.model`.

── LO QUE CAMBIA RESPECTO AL ESQUELETO ──────────────────────────────────────────

  - **Las cajas no salen de un guion.** El catálogo y el nivel las generan; la escena no
    sabe dónde va a acabar ninguna, y eso es justo lo que la diferencia del repo
    guionizado. `layer_heights`, `layer_base_z` y `planned_pose` no existen: lo que las
    sustituye es `PlacementPlan.position`, que decide el planificador.
  - **Hay tres fuentes, no una.** El nivel dice si la carga espera en la mesa, llega por
    la cinta o viene apilada en un remolque, y la escena monta el mobiliario de esa. Una
    mesa auxiliar vacía acompaña a las tres para poder apartar bultos. Ver
    `src/cell/conveyor.py`.
  - **Cada tipo de paquete lleva `cog_offset`**, y se monta como un `<inertial>`
    explícito dentro del body, no como `pos` del geom: así el cartón se ve centrado y
    sólo la inercia está descentrada, que es lo que pasa de verdad cuando el contenido se
    asienta a un lado. Decidido aquí y en un solo sitio.

`build_scene` devuelve la escena con la física ya asentada. Medir antes de eso es medir
cajas todavía cayendo.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.cell import TCP_SITE, add_table, lookat_quat, plant

REPO = Path(__file__).resolve().parents[2]

# Clases de colisión. El brazo es la 2 y los paquetes la 1, así que el brazo choca con
# la celda y con el mobiliario pero NO con el cartón que cuelga de sus propias ventosas.
# Sin esta separación, el paquete recién sellado empuja al brazo que lo lleva.
PACKAGE_CONTACT = 1
ARM_CONTACT = 2
SOLID = 3                       # sólido para los dos

# Grupo de visualización de los paquetes, y por tanto el atajo del visor que los esconde:
# MuJoCo enciende y apaga el grupo N de geoms con la tecla N (0-5), así que `1` quita
# los bultos y deja ver lo que tapan, como la rejilla del mapa de alturas (`m`). No hace
# falta `key_callback`: si `palletize.py` atendiera también el `1`, las dos pulsaciones
# se anularían. Nadie más usa el 1 —el 0 es la celda, el 2 las mallas del UR10e y el 3
# sus colisiones— y NO puede pasar del 2: los renderers fuera de pantalla (fotos y
# cámaras de profundidad) usan la `MjvOption` por defecto, que sólo pinta los grupos
# 0-2, y un bulto en el 3 desaparecería de la percepción sin dar error.
PACKAGE_GROUP = 1


@dataclass(frozen=True)
class Box:
    """Un paquete de la escena: lo que es de verdad, y dónde vive en el modelo.

    Esto es la VERDAD de la simulación. `src/vision/` tiene que llegar a `dims_m`,
    `mass_kg` y `cog_offset_m` midiendo; leerlos de aquí es lo que hace el stub-oráculo,
    y por eso vive en un fichero con el nombre puesto.
    """

    index: int
    package_id: str
    type_name: str
    dims_m: tuple[float, float, float]      # COMPLETAS, no semiejes
    mass_kg: float
    cog_offset_m: tuple[float, float, float]
    friction: tuple[float, float, float]
    rgba: tuple[float, float, float, float]

    @property
    def body(self) -> str:
        return f"package_{self.index}"

    @property
    def joint(self) -> str:
        return f"package_joint_{self.index}"

    @property
    def geom(self) -> str:
        return f"package_geom_{self.index}"

    @property
    def footprint_area(self) -> float:
        return self.dims_m[0] * self.dims_m[1]


@dataclass(frozen=True)
class Level:
    """Un nivel del catálogo de `configs/pallet.yaml`.

    El `id` es lo que se sube a `episodes.level`, y la DECENA es la tarea: 1x mesa,
    2x cinta, 3x camión, 4x ajetreo. Ver la cabecera de `configs/pallet.yaml` para por
    qué se codifica así y no con un campo aparte.

    `task` y `source` NO son lo mismo, aunque en los niveles 1x-3x coincidan. `source`
    es de dónde salen los bultos —mesa, cinta o camión, y eso decide qué `Supply` se
    monta—; `task` es qué se está midiendo, y es lo que viaja a `episodes.task`. Los
    niveles de ajetreo cogen de una fuente cualquiera y lo que miden es otra cosa: si
    la pila aguanta el transporte. Por eso `task` se declara aparte y por omisión vale
    la fuente, que es lo que hacían los niveles de antes.
    """

    id: int
    source: str
    name: str
    n_packages: int
    types: tuple[str, ...]
    yaw_jitter_deg: float = 0.0
    pos_jitter_m: float = 0.0
    cog: str = "catalogue"           # centred | catalogue | adversarial
    loader_jitter: bool = False
    decor: str = "cell"              # cell | plant — ver DECORS
    task: str = ""                   # vacío = la fuente; ver TASKS

    @property
    def task_index(self) -> int:
        return self.id // 10

    @property
    def shakes(self) -> bool:
        """Si al acabar de apilar hay que someter el palé al ensayo de transporte."""
        return self.task == "ajetreo"


SOURCES = ("table", "conveyor", "truck")

# Qué se mide en el episodio, y con ello la decena de su `id`. Es un vocabulario CERRADO
# y lo valida además el SDK: `theker_telemetry` compara `task` contra su propio
# `frozenset` y la base tiene un CHECK, así que un nombre nuevo aquí no basta —hay que
# añadirlo también en Platform antes de poder subir nada con él—.
TASKS = ("table", "conveyor", "truck", "ajetreo")
TASK_DECADE = {"table": 1, "conveyor": 2, "truck": 3, "ajetreo": 4}

# El decorado del nivel, y con él su iluminación: los dos van juntos porque una nave con
# las luces de un plató no es una nave. `cell` es la valla de seguridad y las marcas de
# suelo de siempre; `plant` es la planta industrial de `src/cell/plant.py`.
#
# Es un vocabulario CERRADO, como `source`, `cog`, `events.kind` y `snapshots.view`. Un
# nombre inventado aquí no daría error donde se escribe —`row.get` se lo tragaría— y la
# escena saldría con el decorado por defecto sin decir nada.
DECORS = ("cell", "plant")

# Cada cuánto se refresca el visor, en segundos de reloj de pared. Ver `sync_viewer`.
FRAME_SECONDS = 1.0 / 60.0


def load_configs(root: Path | None = None) -> dict:
    """Los dos YAML en un solo diccionario.

    `pallet.yaml` pisa a `scene.yaml` bloque a bloque, no fichero a fichero: la tarea
    sólo redefine lo que necesita —hoy `motion`— y hereda el resto del banco de trabajo.
    """
    root = Path(root) if root is not None else REPO
    scene = yaml.safe_load((root / "configs" / "scene.yaml").read_text(encoding="utf-8"))
    pallet = yaml.safe_load((root / "configs" / "pallet.yaml").read_text(encoding="utf-8"))

    cfg = dict(scene)
    for key, value in pallet.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    cfg["_root"] = root
    return cfg


def levels(cfg: dict) -> dict[int, Level]:
    """Los niveles declarados, por id."""
    out: dict[int, Level] = {}
    for row in cfg["levels"]:
        level = Level(
            id=int(row["id"]),
            source=str(row["source"]),
            name=str(row["name"]),
            n_packages=int(row["n_packages"]),
            types=tuple(row["types"]),
            yaw_jitter_deg=float(row.get("yaw_jitter_deg", 0.0)),
            pos_jitter_m=float(row.get("pos_jitter_m", 0.0)),
            cog=str(row.get("cog", "catalogue")),
            loader_jitter=bool(row.get("loader_jitter", False)),
            decor=str(row.get("decor", "cell")),
            task=str(row.get("task") or row["source"]),
        )
        if level.source not in SOURCES:
            raise ValueError(f"nivel {level.id}: fuente desconocida {level.source!r}")
        if level.decor not in DECORS:
            raise ValueError(f"nivel {level.id}: decorado desconocido {level.decor!r}")
        if level.task not in TASKS:
            raise ValueError(f"nivel {level.id}: tarea desconocida {level.task!r}")
        if level.task_index != TASK_DECADE[level.task]:
            raise ValueError(
                f"nivel {level.id}: la decena no casa con la tarea {level.task!r}. "
                f"Ver la cabecera de configs/pallet.yaml."
            )
        if level.id in out:
            raise ValueError(f"nivel {level.id} declarado dos veces")
        out[level.id] = level
    return out


def level_for(cfg: dict, level_id: int | None = None) -> Level:
    """El nivel pedido, o el que declara `episode.level`."""
    table = levels(cfg)
    wanted = int(cfg["episode"]["level"] if level_id is None else level_id)
    if wanted not in table:
        raise ValueError(
            f"nivel {wanted} no declarado. Los que hay: {sorted(table)}. "
            f"Añadir uno es añadir una fila a configs/pallet.yaml."
        )
    return table[wanted]


# ─────────────────────────────────────────────────────────────────────────────
# El catálogo: de qué está hecha la carga de este episodio
# ─────────────────────────────────────────────────────────────────────────────

def _cog_for(level: Level, dims, catalogue_offset, rng) -> tuple[float, float, float]:
    """Dónde le toca el centro de gravedad a este paquete, según el nivel.

    `adversarial` no es ruido: empuja el CoG al borde del sobre permitido y alterna el
    signo, para que el montón tenga desequilibrios que se suman en vez de cancelarse.
    Es lo que separa "mucha superficie apoyada" de "estable".
    """
    if level.cog == "centred":
        return (0.0, 0.0, 0.0)
    if level.cog == "catalogue":
        return tuple(float(value) for value in catalogue_offset)
    if level.cog != "adversarial":
        raise ValueError(f"nivel {level.id}: `cog` desconocido {level.cog!r}")
    fraction = 0.25                       # el tope del sobre del generador
    signs = rng.choice((-1.0, 1.0), size=3)
    return (
        float(signs[0] * fraction * dims[0] / 2),
        float(signs[1] * fraction * dims[1] / 2),
        float(-abs(signs[2]) * fraction * dims[2] / 2),   # el contenido se asienta abajo
    )


def _sample_random_type(cfg: dict, rng) -> tuple[tuple[float, float, float], float]:
    """Una caja sorteada del sobre de `configs/pallet.yaml: generator`.

    La masa NO se sortea: es densidad x volumen. Es lo que hace que un bulto grande y
    ligero y otro pequeño y denso sean problemas distintos para el planificador en vez
    de dos números sin relación.
    """
    env = cfg["generator"]
    for _ in range(10_000):
        length = float(rng.uniform(*env["length"]))
        width = float(rng.uniform(*env["width"]))
        height = float(rng.uniform(*env["height"]))
        if width > length:
            length, width = width, length
        volume = length * width * height
        if not env["volume"][0] <= volume <= env["volume"][1]:
            continue
        if length > env["max_aspect_ratio"] * height:
            continue
        mass = float(rng.uniform(*env["density"])) * volume
        if not env["mass"][0] <= mass <= env["mass"][1]:
            continue
        return (round(length, 3), round(width, 3), round(height, 3)), round(mass, 3)
    raise RuntimeError("el sobre del generador no deja salir ninguna caja: revisa `generator`")


def build_catalogue(cfg: dict, level: Level, seed: int) -> list[Box]:
    """Los paquetes de este episodio: qué son y en qué orden se sortearon.

    El sistema NO sabe lo que viene. Lo que la semilla fija es la carga, no el plan: dos
    ejecuciones de la misma semilla ven los mismos bultos, y eso es lo que hace que una
    mejora del planificador se pueda medir contra la anterior.
    """
    rng = np.random.default_rng(seed)
    catalogue = cfg["packages"]
    palette = [tuple(spec["rgba"]) for spec in catalogue.values()]

    # Con varios tipos declarados se BARAJAN y se recorren, en vez de sortear con
    # reemplazo: un nivel que dice "mezcla de seis" y sale con tres repetidos y tres sin
    # aparecer no está probando la mezcla que declara. La semilla sigue decidiendo el
    # orden, que es lo que el sistema no sabe de antemano.
    order: list[str] = []
    if level.types != ("random",):
        while len(order) < level.n_packages:
            batch = list(level.types)
            rng.shuffle(batch)
            order.extend(batch)

    boxes: list[Box] = []
    for index in range(level.n_packages):
        if level.types == ("random",):
            dims, mass = _sample_random_type(cfg, rng)
            type_name = "random"
            catalogue_cog = (0.0, 0.0, 0.0)
            friction = (0.95, 0.01, 0.001)
            rgba = palette[index % len(palette)]
        else:
            type_name = order[index]
            spec = catalogue[type_name]
            dims = tuple(float(value) for value in spec["dims"])
            mass = float(spec["mass"])
            catalogue_cog = tuple(float(value) for value in spec["cog_offset"])
            friction = tuple(float(value) for value in spec["friction"])
            rgba = tuple(float(value) for value in spec["rgba"])

        boxes.append(
            Box(
                index=index,
                package_id=f"{type_name}-{index:02d}",
                type_name=type_name,
                dims_m=dims,
                mass_kg=mass,
                cog_offset_m=_cog_for(level, dims, catalogue_cog, rng),
                friction=friction,
                rgba=rgba,
            )
        )
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# El modelo: MJCF por plantilla
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeldPackage:
    """Un paquete que cuelga rígidamente de la herramienta, en el frame de ésta.

    Sellar una ventosa es una unión rígida, y modelarla reparentando el cuerpo no es
    sólo más simple: una `weld` sesga el par que informa el sensor de muñeca, lo
    bastante a estas masas como para que medir el centro de gravedad no signifique nada.
    Reparentar quita la restricción del todo, así que no queda nada que contabilizar mal.
    """

    index: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


_MATERIALS = """
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.16 0.18 0.21"
             rgb2="0.22 0.24 0.27" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.08"/>
    <material name="black" rgba="0.033 0.033 0.033 1" specular="0.5" shininess="0.25"/>
    <material name="jointgray" rgba="0.278 0.278 0.278 1" specular="0.5" shininess="0.25"/>
    <material name="linkgray" rgba="0.82 0.82 0.82 1" specular="0.5" shininess="0.25"/>
    <material name="urblue" rgba="0.49 0.678 0.8 1" specular="0.5" shininess="0.25"/>
    <material name="tool" rgba="0.10 0.13 0.16 1"/>
    <material name="aluminium" rgba="0.62 0.66 0.69 1" metallic="0.35" roughness="0.28"/>
    <material name="onrobot_orange" rgba="0.94 0.35 0.07 1" roughness="0.42"/>
    <material name="pallet_wood" rgba="0.62 0.39 0.18 1" roughness="0.82"/>
    <material name="pallet_wood_dark" rgba="0.45 0.27 0.12 1" roughness="0.88"/>
    <material name="frame" rgba="0.16 0.19 0.22 1" metallic="0.55" roughness="0.30"/>
    <material name="roller" rgba="0.31 0.34 0.36 1" metallic="0.75" roughness="0.22"/>
    <material name="belt" rgba="0.14 0.15 0.17 1" roughness="0.90"/>
    <material name="fence" rgba="0.12 0.14 0.15 1" metallic="0.45" roughness="0.45"/>
    <material name="safety_yellow" rgba="0.95 0.68 0.04 1" roughness="0.55"/>
    <material name="truck_deck" rgba="0.34 0.28 0.22 1" roughness="0.85"/>
    <material name="truck_deck_line" rgba="0.24 0.19 0.15 1" roughness="0.85"/>
    <material name="truck_body" rgba="0.87 0.88 0.90 1" roughness="0.45"/>
    <material name="truck_rib" rgba="0.78 0.79 0.82 1" roughness="0.45"/>
    <material name="truck_door" rgba="0.82 0.84 0.87 1" roughness="0.45"/>
    <material name="truck_frame" rgba="0.36 0.39 0.43 1" metallic="0.5" roughness="0.35"/>"""

UR10E_MESHES = (
    "base_0", "base_1", "shoulder_0", "shoulder_1", "shoulder_2",
    "upperarm_0", "upperarm_1", "upperarm_2", "upperarm_3",
    "forearm_0", "forearm_1", "forearm_2", "forearm_3",
    "wrist1_0", "wrist1_1", "wrist1_2",
    "wrist2_0", "wrist2_1", "wrist2_2", "wrist3",
)


def mesh_dir(cfg: dict) -> Path:
    """Dónde están las mallas del UR10e. Las clona `scripts/setup.sh`."""
    return Path(cfg["_root"]) / cfg["robot"]["model"] / "assets"


def _visuals(simplified: bool, *items: tuple[str, str]) -> str:
    if simplified:
        return ""
    return "\n".join(
        f'<geom mesh="{mesh}" material="{material}" class="robot_visual"/>'
        for mesh, material in items
    )


def _collision(simplified: bool, material: str) -> str:
    """Las primitivas de colisión: visibles en modo simple, invisibles con mallas.

    Las mallas del UR10e mejoran el render, pero las colisiones siguen siendo primitivas
    convexas para que la simulación sea rápida y estable. Son dos geometrías distintas a
    propósito, y por eso se pueden enseñar por separado.
    """
    return f'material="{material}"' if simplified else 'rgba="0 0 0 0" group="3"'


def _package_xml(box: Box, held: HeldPackage | None, simplified: bool) -> str:
    """El cuerpo de un paquete: libre en la escena, o colgando de la herramienta."""
    width, depth, height = box.dims_m
    ix = box.mass_kg * (depth**2 + height**2) / 12
    iy = box.mass_kg * (width**2 + height**2) / 12
    iz = box.mass_kg * (width**2 + depth**2) / 12
    friction = " ".join(str(value) for value in box.friction)
    rgba = " ".join(str(value) for value in box.rgba)

    # El CoG se monta como `<inertial>` explícito, no como `pos` del geom: el cartón se
    # ve centrado y sólo la inercia está descentrada, que es lo que pasa de verdad.
    inertial = (
        f'<inertial pos="{box.cog_offset_m[0]:.6f} {box.cog_offset_m[1]:.6f} '
        f'{box.cog_offset_m[2]:.6f}" mass="{box.mass_kg}" '
        f'diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"/>'
    )
    geom = (
        f'<geom name="{box.geom}" type="box" '
        f'size="{width / 2:.6f} {depth / 2:.6f} {height / 2:.6f}" density="0" '
        f'friction="{friction}" contype="{PACKAGE_CONTACT}" conaffinity="{PACKAGE_CONTACT}" '
        f'rgba="{rgba}" group="{PACKAGE_GROUP}"/>'
    )
    trim = ""
    if not simplified:
        tape = min(width * 0.10, 0.045)
        label_w = min(width * 0.27, 0.105)
        label_h = min(height * 0.28, 0.055)
        trim = f"""
      <geom type="box" pos="0 0 {height / 2 + 0.001:.6f}"
            size="{tape / 2:.6f} {depth / 2:.6f} 0.0015" rgba="0.73 0.61 0.40 1"
            contype="0" conaffinity="0" group="{PACKAGE_GROUP}"/>
      <geom type="box" pos="{width * 0.18:.6f} {-depth / 2 - 0.001:.6f} {height * 0.08:.6f}"
            size="{label_w / 2:.6f} 0.0015 {label_h / 2:.6f}" rgba="0.94 0.95 0.91 1"
            contype="0" conaffinity="0" group="{PACKAGE_GROUP}"/>"""

    if held is not None and held.index == box.index:
        position = " ".join(f"{value:.6f}" for value in held.position)
        quaternion = " ".join(f"{value:.6f}" for value in held.quaternion)
        return f"""
    <body name="{box.body}" pos="{position}" quat="{quaternion}">
      {inertial}
      {geom}{trim}
    </body>"""

    parked = -3.0 - box.index * 0.7
    return f"""
    <body name="{box.body}" pos="{parked:.3f} 0 {height / 2 + 0.002:.6f}">
      <freejoint name="{box.joint}"/>
      {inertial}
      {geom}{trim}
    </body>"""


def _weld_xml(box: Box) -> str:
    """La unión de vacío, inactiva hasta que se sella.

    `relpose` se sustituye por la transformada medida justo antes de agarrar.
    `torquescale` se queda en 1: dieciséis ventosas sobre 264 x 184 mm sujetan un cartón
    plano, y un valor flojo deja que un golpe lo haga girar sobre la herramienta —medido,
    5 Nm lo vuelca del todo a 0.15 frente a 1,3 grados aquí.
    """
    return (
        f'<weld name="suction_{box.index}" body1="tool" body2="{box.body}" '
        'relpose="0 0 0 1 0 0 0" torquescale="1" active="false"/>'
    )


def _cups_xml(cfg: dict, simplified: bool) -> str:
    """Las dieciséis ventosas, en su rejilla del frame de la herramienta."""
    vac = cfg["vacuum"]
    radius = vac["cup_radius"]
    flange_to_cup = vac["body_dims"][2]
    lip_z = flange_to_cup - radius * 0.3
    stem_z = lip_z - 0.014
    out: list[str] = []
    for column in range(vac["columns"]):
        x = (column - (vac["columns"] - 1) / 2) * vac["pitch_x"]
        for row in range(vac["rows"]):
            y = (row - (vac["rows"] - 1) / 2) * vac["pitch_y"]
            if simplified:
                out.append(
                    f'<geom name="cup_{column}_{row}" type="cylinder" '
                    f'pos="{x:.6f} {y:.6f} {lip_z:.6f}" size="{radius} 0.012" '
                    'rgba="0.10 0.85 0.90 1" contype="0" conaffinity="0"/>'
                )
            else:
                out.append(
                    f'<geom type="cylinder" pos="{x:.6f} {y:.6f} {stem_z:.6f}" '
                    'size="0.007 0.010" material="aluminium" contype="0" conaffinity="0"/>'
                    f'<geom name="cup_{column}_{row}" type="cylinder" '
                    f'pos="{x:.6f} {y:.6f} {lip_z:.6f}" size="{radius} 0.006" '
                    'rgba="0.08 0.12 0.14 1" contype="0" conaffinity="0"/>'
                )
    return "\n".join(out)


def _ur10e_xml(cfg: dict, cups: str, carried: str, simplified: bool) -> str:
    """UR10e con la cinemática, masas e inercias oficiales del Menagerie.

    El site `tcp` es el frame que usa la IK, y está en la cara de las ventosas, no en la
    brida: lo que tiene que llegar al sitio es lo que agarra.
    """
    robot = cfg["robot"]
    vac = cfg["vacuum"]
    body_x = vac["body_dims"][0] / 2
    body_y = vac["body_dims"][1] / 2
    flange_to_cup = vac["body_dims"][2]
    base_pos = " ".join(str(value) for value in robot["base_pos"])
    base_quat = " ".join(str(value) for value in robot["base_quat"])

    if simplified:
        tool = f"""
                    <geom type="box" size="{body_x} {body_y} 0.026" pos="0 0 0.040"
                          material="tool" contype="0" conaffinity="0"/>
                    {cups}"""
    else:
        tool = f"""
                    <geom type="cylinder" pos="0 0 0.007" size="0.0315 0.007"
                          material="jointgray" contype="0" conaffinity="0"/>
                    <geom name="vgp20_body" type="box" pos="0 0 0.040"
                          size="{body_x} {body_y} 0.026" material="tool"
                          contype="0" conaffinity="0"/>
                    <geom type="box" pos="0 0 0.018"
                          size="{body_x - 0.014} {body_y - 0.014} 0.006" material="jointgray"
                          contype="0" conaffinity="0"/>
                    <geom type="box" pos="0 {body_y + 0.0015} 0.040" size="0.055 0.0015 0.014"
                          material="onrobot_orange" contype="0" conaffinity="0"/>
                    {cups}"""

    return f"""
    <body name="ur10e_base" pos="{base_pos}" quat="{base_quat}" childclass="ur10e">
      <inertial mass="4.0" pos="0 0 0" diaginertia="0.006106 0.006106 0.01125"/>
      {_visuals(simplified, ("base_0", "black"), ("base_1", "jointgray"))}
      <geom type="cylinder" size="0.095 0.075" pos="0 0 -0.02" {_collision(simplified, "jointgray")}/>
      <body name="shoulder_link" pos="0 0 0.181">
        <inertial pos="0 0 0" mass="7.778" diaginertia="0.0314743 0.0314743 0.0218756"/>
        <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>
        {_visuals(simplified, ("shoulder_0", "urblue"), ("shoulder_1", "black"), ("shoulder_2", "jointgray"))}
        <geom type="cylinder" size="0.088 0.09" {_collision(simplified, "urblue")}/>
        <body name="upper_arm_link" pos="0 0.176 0" quat="1 0 1 0">
          <inertial pos="0 0 0.3065" mass="12.93" diaginertia="0.423074 0.423074 0.036366"/>
          <joint name="shoulder_lift_joint" class="size4"/>
          {_visuals(simplified, ("upperarm_0", "black"), ("upperarm_1", "jointgray"), ("upperarm_2", "urblue"), ("upperarm_3", "linkgray"))}
          <geom type="cylinder" size="0.084 0.09" pos="0 -0.05 0" quat="1 1 0 0" {_collision(simplified, "jointgray")}/>
          <geom type="capsule" size="0.068" fromto="0 0 0.05 0 0 0.59" {_collision(simplified, "urblue")}/>
          <body name="forearm_link" pos="0 -0.137 0.613">
            <inertial pos="0 0 0.2855" mass="3.87" diaginertia="0.11059 0.11059 0.010884"/>
            <joint name="elbow_joint" class="size3_limited"/>
            {_visuals(simplified, ("forearm_0", "urblue"), ("forearm_1", "black"), ("forearm_2", "jointgray"), ("forearm_3", "linkgray"))}
            <geom type="cylinder" size="0.074 0.075" pos="0 0.07 0" quat="1 1 0 0" {_collision(simplified, "jointgray")}/>
            <geom type="capsule" size="0.055" fromto="0 0 0.05 0 0 0.55" {_collision(simplified, "linkgray")}/>
            <body name="wrist_1_link" pos="0 0 0.571" quat="1 0 1 0">
              <inertial pos="0 0.135 0" quat="0.5 0.5 -0.5 0.5" mass="1.96"
                        diaginertia="0.0055125 0.0051083 0.0051083"/>
              <joint name="wrist_1_joint" class="size2"/>
              {_visuals(simplified, ("wrist1_0", "black"), ("wrist1_1", "urblue"), ("wrist1_2", "jointgray"))}
              <geom type="cylinder" size="0.058 0.07" pos="0 0.06 0" quat="1 1 0 0" {_collision(simplified, "jointgray")}/>
              <body name="wrist_2_link" pos="0 0.135 0">
                <inertial pos="0 0 0.12" quat="0.5 0.5 -0.5 0.5" mass="1.96"
                          diaginertia="0.0055125 0.0051083 0.0051083"/>
                <joint name="wrist_2_joint" axis="0 0 1" class="size2"/>
                {_visuals(simplified, ("wrist2_0", "black"), ("wrist2_1", "urblue"), ("wrist2_2", "jointgray"))}
                <geom type="cylinder" size="0.052 0.065" pos="0 0 0.05" {_collision(simplified, "urblue")}/>
                <body name="wrist_3_link" pos="0 0 0.12">
                  <inertial pos="0 0.092 0" quat="0 1 -1 0" mass="0.202"
                            diaginertia="0.0002045 0.0001443 0.0001443"/>
                  <joint name="wrist_3_joint" class="size2"/>
                  {_visuals(simplified, ("wrist3", "linkgray"))}
                  <geom type="cylinder" size="0.050 0.075" pos="0 0.055 0" quat="1 1 0 0" {_collision(simplified, "jointgray")}/>
                  <body name="tool" pos="0 0.10 0" quat="-1 1 0 0">
                    <inertial pos="0 0 0.038" mass="{vac['mass']}"
                              diaginertia="0.0081 0.0158 0.0221"/>
                    <!-- Sensor de fuerza/par de muñeca. En el origen de `tool`, cuyo
                         padre es `wrist_3_link`, lee el wrench que cruza la brida:
                         herramienta más lo que el vacío esté sujetando. -->
                    <site name="ft_site" pos="0 0 0" size="0.006" rgba="0 0 0 0"/>
                    {tool}
                    <site name="{TCP_SITE}" pos="0 0 {flange_to_cup}" size="0.008" rgba="1 0 0 0"/>
                    {carried}
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _pallet_xml(cfg: dict, simplified: bool) -> str:
    """Europalé como cuerpo dinámico, sujeto al mundo mientras el robot trabaja.

    Los geoms visuales no llevan masa, así que la inercia que se mueve es la del palé de
    25 kg y no la suma de las tablas.
    """
    pal = cfg["pallet"]
    width, depth = pal["dims"]
    height = pal["deck_thickness"]
    center_x, center_y = pal["center"]
    mass = 25.0                      # europalé EUR-1
    ixx = mass * (depth**2 + height**2) / 12
    iyy = mass * (width**2 + height**2) / 12
    izz = mass * (width**2 + depth**2) / 12
    rgba = " ".join(str(value) for value in pal["rgba"])

    if simplified:
        geoms = (
            f'<geom name="pallet" type="box" pos="0 0 {height / 2}" '
            f'size="{width / 2} {depth / 2} {height / 2}" density="0" rgba="{rgba}" '
            f'friction="1.1 0.01 0.001" conaffinity="{SOLID}"/>'
        )
    else:
        boards = "\n".join(
            f'<geom type="box" pos="0 {offset} {height - 0.011:.4f}" '
            f'size="{width / 2} 0.055 0.011" material="pallet_wood" density="0" '
            'contype="0" conaffinity="0"/>'
            for offset in (-0.335, -0.17, 0.0, 0.17, 0.335)
        )
        runners = "\n".join(
            f'<geom type="box" pos="0 {offset} 0.011" size="{width / 2} 0.048 0.011" '
            'material="pallet_wood_dark" density="0" contype="0" conaffinity="0"/>'
            for offset in (-0.33, 0.0, 0.33)
        )
        blocks = "\n".join(
            f'<geom type="box" pos="{x} {y} {height / 2}" size="0.075 0.060 0.038" '
            'material="pallet_wood_dark" density="0" contype="0" conaffinity="0"/>'
            for x in (-0.49, 0.0, 0.49)
            for y in (-0.33, 0.0, 0.33)
        )
        geoms = f"""
      <geom name="pallet" type="box" pos="0 0 {height / 2}"
            size="{width / 2} {depth / 2} {height / 2}" density="0" rgba="0 0 0 0"
            friction="1.1 0.01 0.001" conaffinity="{SOLID}"/>
      {boards}
      {runners}
      {blocks}"""

    return f"""
    <body name="pallet" pos="{center_x} {center_y} 0">
      <inertial pos="0 0 {height / 2}" mass="{mass}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>
      <joint name="pallet_x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="pallet_y" type="slide" axis="0 1 0" damping="2"/>
      <joint name="pallet_z" type="slide" axis="0 0 1" damping="2"/>
      <joint name="pallet_rx" type="hinge" axis="1 0 0" damping="1"
             limited="true" range="-0.0001 0.0001"/>
      <joint name="pallet_ry" type="hinge" axis="0 1 0" damping="1"
             limited="true" range="-0.0001 0.0001"/>
      {geoms}
    </body>"""


def _stability_beam_xml(cfg: dict) -> str:
    """Cresta del ensayo final, aparcada bajo el suelo mientras se paletiza."""
    width, depth = (float(value) for value in cfg["pallet"]["dims"])
    test = cfg["stability_test"]
    length = max(width, depth) / 2 + 0.08
    return f"""
    <body name="stability_beam" mocap="true" pos="0 0 -1">
      <geom name="stability_beam" type="box"
            size="{length} {float(test['beam_width']) / 2} {float(test['beam_height']) / 2}"
            rgba="0.78 0.28 0.14 0" friction="1.2 0.02 0.001"
            density="0" contype="1" conaffinity="1"/>
    </body>"""


def lane_for(cfg: dict, source: str) -> dict | None:
    """El carril por el que llegan los bultos, o `None` si la fuente no tiene.

    Mesa y cinta comparten mecánica y se diferencian en dónde ACABA la banda: la de la
    cinta entrega al brazo y la de la mesa entrega a la mesa. El camión no tiene carril
    —la carga ya viene apilada en el remolque— y por eso tampoco tiene caja negra.
    """
    if source == "conveyor":
        belt = dict(cfg["conveyor"])
        belt["station"] = tuple(float(v) for v in belt["station"])
        return belt
    if source != "table":
        return None
    # La banda de mesa acaba en el canto de la mesa y comparte su carril y su cota, para
    # que el bulto cruce sin escalón. Su centro se deriva: no hay dos números que cuadrar.
    table = cfg["table"]
    feeder = dict(cfg["feeder"])
    table_x, table_y = (float(v) for v in table["center"])
    edge = table_x - float(table["size"][0])
    length = float(feeder["dims"][0])
    feeder["center"] = (edge - length / 2, table_y)
    feeder["station"] = (table_x, table_y)       # se entrega al CENTRO de la mesa
    return feeder


def parking_grid(cfg: dict, boxes: list[Box], lane: dict) -> list[tuple[float, float]]:
    """Dónde espera cada bulto dentro de la caja negra, en rejilla y sin tocarse.

    Antes era una fila —`x = -3 - i*0.7`— que con treinta bultos medía 21 m y se veía
    salir de la escena. La rejilla ocupa lo mismo en superficie y cabe en un cerramiento.
    """
    black = cfg["black_box"]
    pitch_x, pitch_y = (float(v) for v in black["pitch"])
    columns = max(1, int(black["columns"]))
    lane_x = (float(lane["center"][0]) - float(lane["dims"][0]) / 2
              - float(cfg["elevator"]["depth"]) - float(black["gap"]))
    lane_y = float(lane["center"][1])
    slots = []
    for index in range(len(boxes)):
        row, column = divmod(index, columns)
        slots.append((
            lane_x - pitch_x / 2 - row * pitch_x,
            lane_y + (column - (columns - 1) / 2) * pitch_y,
        ))
    return slots


def _black_box_xml(cfg: dict, boxes: list[Box], lane: dict) -> str:
    """El cerramiento opaco, ajustado a la rejilla de espera que tenga que tapar."""
    black = cfg["black_box"]
    slots = parking_grid(cfg, boxes, lane)
    pitch_x, pitch_y = (float(v) for v in black["pitch"])
    margin, wall = float(black["margin"]), float(black["wall"])
    height, rgba = float(black["height"]), " ".join(str(v) for v in black["rgba"])

    xs = [x for x, _ in slots]
    ys = [y for _, y in slots]
    x0, x1 = min(xs) - pitch_x / 2 - margin, max(xs) + pitch_x / 2 + margin
    y0, y1 = min(ys) - pitch_y / 2 - margin, max(ys) + pitch_y / 2 + margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_x, half_y = (x1 - x0) / 2, (y1 - y0) / 2

    # Cerramiento sin suelo; delante sólo queda el hueco de entrega a la banda.
    # Una fachada entera abierta dejaba ver los cartones esperando en el suelo.
    panels = [
        (cx - half_x - wall, cy, height / 2, wall, half_y + wall, height / 2),   # fondo
        (cx, cy - half_y - wall, height / 2, half_x, wall, height / 2),          # lateral
        (cx, cy + half_y + wall, height / 2, half_x, wall, height / 2),          # lateral
        (cx, cy, height + wall, half_x + wall, half_y + wall, wall),             # techo
    ]
    mouth_half = float(lane["dims"][1]) / 2 + 0.02
    mouth_y = float(lane["center"][1])
    mouth_bottom = float(lane["height"])
    mouth_top = mouth_bottom + max(float(box.dims_m[2]) for box in boxes) + 0.05
    front_x = x1 - wall / 2
    panels += [
        (front_x, cy, mouth_bottom / 2, wall / 2, half_y, mouth_bottom / 2),
        (front_x, cy, (height + mouth_top) / 2, wall / 2, half_y, (height - mouth_top) / 2),
    ]
    for low, high in ((y0, mouth_y - mouth_half), (mouth_y + mouth_half, y1)):
        if high > low:
            panels.append((front_x, (low + high) / 2, height / 2,
                           wall / 2, (high - low) / 2, height / 2))
    return "\n".join(
        f'    <geom name="black_box_{index}" type="box" pos="{px:.4f} {py:.4f} {pz:.4f}" '
        f'size="{sx:.4f} {sy:.4f} {sz:.4f}" rgba="{rgba}" contype="0" conaffinity="0"/>'
        for index, (px, py, pz, sx, sy, sz) in enumerate(panels)
    )


def _belt_xml(cfg: dict, simplified: bool, lane: dict | None = None) -> str:
    """La banda transportadora. La superficie es estática; lo que se mueve es la carga.

    MuJoCo no tiene primitiva de cinta. De las dos opciones razonables —una banda con
    `slide` y actuador de velocidad, o aplicar velocidad a los cuerpos en contacto— se
    usa la segunda, que es la que `src/cell/conveyor.py` implementa: una banda con
    `slide` de 1,4 m se queda sin recorrido a la mitad del trayecto y habría que
    reponerla, y reponerla es teletransportar por la puerta de atrás.

    La fricción es alta a propósito: es lo que arrastra el cartón sin que patine.
    """
    belt = cfg["conveyor"] if lane is None else lane
    x, y = belt["center"]
    length, width = belt["dims"]
    top = belt["height"]
    friction = " ".join(str(value) for value in belt["friction"])
    # DOS tramos, que es como se monta una línea de verdad: uno sale del ascensor y el
    # otro entrega. Se tocan exactamente en el centro —no hay junta por la que colarse— y
    # el arrastre no se entera de que son dos, porque `Belt._drive` mira la ALTURA a la
    # que va el cartón, no sobre qué geometría. Por eso partirla es gratis aquí y por eso
    # el bulto también cruza a la mesa sin escalón.
    half = length / 4
    decks = [
        (f'{x - half:.4f}', 'segment_a'),
        (f'{x + half:.4f}', 'segment_b'),
    ]
    deck = "\n".join(
        f'    <geom name="belt_{name}" type="box" pos="{cx} {y} {top - 0.02:.4f}" '
        f'size="{half:.4f} {width / 2} 0.02" friction="{friction}" '
        f'conaffinity="{SOLID}" rgba="{{rgba}}"/>'
        for cx, name in decks
    )
    if simplified:
        return deck.format(rgba="0.14 0.15 0.17 1") + "\n" + _elevator_xml(cfg, belt)

    rollers = "\n".join(
        f'    <geom type="cylinder" pos="{value:.4f} {y} {top - 0.055:.4f}" '
        f'quat="0.7071 0.7071 0 0" size="0.026 {width / 2 - 0.02:.4f}" material="roller" '
        'contype="0" conaffinity="0"/>'
        for value in np.arange(x - length / 2 + 0.08, x + length / 2, 0.16)
    )
    sides = "\n".join(
        f'    <geom type="box" pos="{x} {y + side * (width / 2 + 0.035):.4f} {top - 0.05:.4f}" '
        f'size="{length / 2 + 0.02} 0.035 0.075" material="frame" contype="0" conaffinity="0"/>'
        for side in (-1, 1)
    )
    legs = "\n".join(
        f'    <geom type="box" pos="{x + dx:.4f} {y + dy:.4f} {(top - 0.09) / 2:.4f}" '
        f'size="0.035 0.035 {(top - 0.09) / 2:.4f}" material="frame" contype="0" conaffinity="0"/>'
        for dx in (-length / 2 + 0.10, 0.0, length / 2 - 0.10)
        for dy in (-width / 2, width / 2)
    )
    station_x, station_y = belt["station"]
    marker = (
        f'    <geom type="box" pos="{station_x} {station_y} {top + 0.0005:.4f}" '
        f'size="0.004 {width / 2 - 0.02:.4f} 0.0005" material="safety_yellow" '
        'contype="0" conaffinity="0"/>'
    )
    # La banda visible SÍ va partida y con su hueco: es lo que hace que se lean dos
    # tramos y no una cinta larga. El hueco es sólo visual —la superficie de arrastre de
    # arriba es continua—, así que no hay por dónde caerse.
    seam = 0.012
    surfaces = "\n".join(
        f'    <geom type="box" pos="{cx} {y} {top - 0.02:.4f}" '
        f'size="{half - seam:.4f} {width / 2} 0.019" material="belt" '
        'contype="0" conaffinity="0"/>'
        for cx, _ in decks
    )
    transfer = "\n".join([
        f'    <geom type="cylinder" pos="{x + dx:.4f} {y} {top - 0.028:.4f}" '
        f'quat="0.7071 0.7071 0 0" size="0.018 {width / 2 - 0.01:.4f}" material="roller" '
        'contype="0" conaffinity="0"/>'
        for dx in (-0.020, 0.020)
    ])
    return deck.format(rgba="0 0 0 0") + "\n" + "\n".join(
        [surfaces, transfer, rollers, sides, legs, marker,
         _elevator_xml(cfg, belt)]
    )


def _elevator_xml(cfg: dict, lane: dict) -> str:
    """Plataforma móvil antes de la banda; `Belt` manda su altura y el contacto eleva la caja."""
    tower = cfg["elevator"]
    length = float(lane["dims"][0])
    top = float(tower["lower_height"])
    depth = float(tower["depth"]) / 2
    # El mismo ancho evita un cambio lateral de apoyo al cruzar a la cinta.
    platform_width = float(lane["dims"][1])
    half_w = platform_width / 2 + float(tower["side"])
    # El canto de salida toca la banda. Debajo de la plataforma no hay banda fija:
    # de lo contrario la caja chocaría con ella al subir desde el suelo.
    x = float(lane["center"][0]) - length / 2 - depth
    y = float(lane["center"][1])
    height = float(tower["height"])
    rgba = " ".join(str(v) for v in tower["rgba"])
    friction = " ".join(str(value) for value in lane["friction"])
    trim = 'contype="0" conaffinity="0"'
    parts = [
        f'    <body name="elevator" mocap="true" pos="{x:.4f} {y:.4f} {top - 0.012:.4f}">'
        f'<geom name="elevator_platform" type="box" size="{depth:.4f} {platform_width / 2:.4f} 0.012" '
        f'material="roller" friction="{friction}" solref="0.004 1" conaffinity="{SOLID}"/></body>',
        f'    <geom name="elevator_header" type="box" pos="{x:.4f} {y:.4f} {height:.4f}" '
        f'size="{depth:.4f} {half_w:.4f} 0.05" material="safety_yellow" {trim}/>',
    ]
    parts += [
        f'    <geom type="box" pos="{x + dx:.4f} {y + dy:.4f} {height / 2:.4f}" '
        f'size="0.025 0.025 {height / 2:.4f}" rgba="{rgba}" {trim}/>'
        for dx in (-depth + 0.025, depth - 0.025)
        for dy in (-half_w + 0.025, half_w - 0.025)
    ]
    parts += [
        f'    <geom type="box" pos="{x:.4f} {y + side * half_w:.4f} {height / 2:.4f}" '
        f'size="0.025 0.02 {height / 2:.4f}" material="roller" {trim}/>'
        for side in (-1, 1)
    ]
    return "\n".join(parts)


def _truck_xml(cfg: dict, simplified: bool) -> str:
    """El remolque arrimado al muelle, visto desde dentro por las puertas abiertas.

    Sólo el piso lleva colisiones de carga; la carrocería es estructura, no decorado: un
    cartón que se va contra ella choca, y el brazo también.

    El lado por el que trabaja el brazo se deja abierto —lonero con la lona recogida—
    porque el brazo cruza esa línea en cada ciclo y un panel ahí sería una pared dentro
    de la que la celda se pasa la ejecución entera. La pared del fondo sí es real, y
    tiene que quedar despejada de todo lo que la celda balancea: pesar un cartón lo saca
    casi un tercio de metro más allá de donde se cogió, así que `aisle` es esa
    basculación, medida en la celda.
    """
    bay = cfg["truck"]
    x0, y0 = bay["origin"]
    x1, y1 = x0 + bay["width"], y0 + bay["depth"]
    deck = bay["floor_height"]
    rear, cab = x1 + 0.08, x0 - 1.30
    far, kerb = y0 - bay["aisle"], y1 + 0.035
    wall = 1.30
    solid = f'conaffinity="{SOLID}" friction="0.7 0.01 0.001"'
    trim = 'contype="0" conaffinity="0"'
    mid_x, mid_y = (cab + rear) / 2, (far + kerb) / 2
    half_x, half_y = (rear - cab) / 2, (kerb - far) / 2

    if simplified:
        return f"""
    <geom name="truck_deck" type="box" pos="{mid_x} {mid_y} {deck / 2}"
          size="{half_x} {half_y} {deck / 2}" rgba="0.26 0.28 0.31 1"
          friction="0.9 0.01 0.001" conaffinity="{SOLID}"/>
    <geom name="truck_far_wall" type="box" pos="{mid_x} {far} {deck + wall / 2}"
          size="{half_x} 0.02 {wall / 2}" rgba="0.80 0.82 0.85 1" {solid}/>"""

    deep = x0 - 0.25
    planks = "\n".join(
        f'    <geom type="box" pos="{mid_x} {value:.4f} {deck + 0.0015}" '
        f'size="{half_x} 0.004 0.0015" material="truck_deck_line" {trim}/>'
        for value in np.linspace(far + 0.10, kerb - 0.10, 7)
    )
    ribs = "\n".join(
        f'    <geom type="box" pos="{value:.4f} {far - 0.022} {deck + wall / 2}" '
        f'size="0.012 0.008 {wall / 2 - 0.02}" material="truck_rib" {trim}/>'
        for value in np.arange(cab + 0.12, rear - 0.05, 0.24)
    )
    return f"""
    <geom name="truck_deck" type="box" pos="{mid_x} {mid_y} {deck / 2}"
          size="{half_x} {half_y} {deck / 2}" rgba="0 0 0 0"
          friction="0.9 0.01 0.001" conaffinity="{SOLID}"/>
    <geom type="box" pos="{mid_x} {mid_y} {deck / 2}"
          size="{half_x} {half_y} {deck / 2 - 0.001}" material="truck_deck" {trim}/>
{planks}
    <geom name="truck_far_wall" type="box" pos="{mid_x} {far - 0.012} {deck + wall / 2}"
          size="{half_x} 0.012 {wall / 2}" material="truck_body" {solid}/>
{ribs}
    <geom name="truck_kerb_wall" type="box" pos="{(cab + deep) / 2} {kerb + 0.012} {deck + wall / 2}"
          size="{(deep - cab) / 2} 0.012 {wall / 2}" material="truck_body" {solid}/>
    <geom name="truck_roof" type="box" pos="{(cab + deep) / 2} {mid_y} {deck + wall + 0.012}"
          size="{(deep - cab) / 2} {half_y + 0.024} 0.012" material="truck_body" {solid}/>
    <geom name="truck_bulkhead" type="box" pos="{cab - 0.012} {mid_y} {deck + wall / 2}"
          size="0.012 {half_y} {wall / 2}" material="truck_rib" {solid}/>
    <geom name="truck_kerb_rail" type="box" pos="{(deep + rear) / 2} {kerb + 0.012} {deck + 0.055}"
          size="{(rear - deep) / 2} 0.012 0.055" material="truck_body" {solid}/>
    <geom type="box" pos="{rear + 0.03} {far - 0.012} {deck + wall / 2}"
          size="0.03 0.026 {wall / 2}" material="truck_frame" {solid}/>
    <!-- Una puerta trasera abierta del todo contra la carrocería; la otra, fuera de plano. -->
    <geom name="truck_door" type="box" pos="{rear - 0.42} {far - 0.06} {deck + wall / 2}"
          size="0.40 0.016 {wall / 2 - 0.01}" material="truck_door" {solid}/>
    <geom type="box" pos="{rear + 0.09} {far + 0.10} {deck / 2}"
          size="0.055 0.075 {deck / 2 + 0.02}" material="black" {solid}/>
    <geom type="box" pos="{rear + 0.09} {kerb - 0.10} {deck / 2}"
          size="0.055 0.075 {deck / 2 + 0.02}" material="black" {solid}/>"""


def _industrial_xml(simplified: bool) -> str:
    """Valla de seguridad y marcas de suelo. Puro decorado: no colisiona con nada."""
    if simplified:
        return ""
    posts = "\n".join(
        f'    <geom type="box" pos="{x} 0.92 0.76" size="0.035 0.035 0.76" '
        'material="safety_yellow" contype="0" conaffinity="0"/>'
        for x in (-1.65, -0.85, -0.05, 0.75, 1.55)
    )
    rails = "\n".join(
        f'    <geom type="box" pos="0 0.92 {z}" size="1.65 0.018 0.012" '
        'material="fence" contype="0" conaffinity="0"/>'
        for z in (0.22, 0.47, 0.72, 0.97, 1.22, 1.47)
    )
    markings = "\n".join(
        f'    <geom type="box" pos="{x:.4f} -0.72 0.003" size="0.11 0.025 0.003" '
        'euler="0 0 -0.55" material="safety_yellow" contype="0" conaffinity="0"/>'
        for x in np.linspace(-0.20, 1.40, 9)
    )
    return f"{posts}\n{rails}\n{markings}"


def _decor_xml(cfg: dict, level: Level, simplified: bool) -> str:
    """El decorado del nivel. Ver `DECORS`."""
    if level.decor == "plant":
        return plant.decor_xml(cfg, simplified)
    return _industrial_xml(simplified)


def _floor_xml(cfg: dict, level: Level) -> str:
    """El suelo.

    La fricción y la clase de contacto son las mismas SIEMPRE: lo único que cambia con el
    decorado es la pinta. Con la nave, además, el plano se recorta a su planta de 8 × 6 m
    y se desplaza con ella, porque un plano de 12 × 12 asoma por fuera de las paredes y
    se ve en la `iso`.
    """
    if level.decor == "plant":
        offset_x, offset_y = (float(value) for value in cfg["plant"]["offset"])
        look = f'pos="{offset_x} {offset_y} 0" size="4 3 0.1" material="plant_concrete"'
    else:
        look = 'size="6 6 0.1" material="floor_mat"'
    return (
        f'    <geom name="floor" type="plane" {look}\n'
        f'          friction="1.0 0.01 0.001" conaffinity="{SOLID}"/>'
    )


def _camera_xml(name: str, spec: dict) -> str:
    quat = lookat_quat(spec["position"], spec["lookat"], spec.get("up", (0.0, 0.0, 1.0)))
    position = " ".join(str(value) for value in spec["position"])
    return (
        f'    <camera name="{name}" pos="{position}" '
        f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
        f'fovy="{spec["fovy"]}"/>'
    )


def _cameras_xml(cfg: dict) -> str:
    """Las vistas de la escena: las cuatro que suben fotos y las de percepción.

    LOS NOMBRES DE `cameras:` SON UN VOCABULARIO CERRADO: `top | side | iso | camera`.
    Con otro nombre el PNG sube a Storage y LUEGO la base rechaza la fila con un 23514:
    la foto queda huérfana y la traza sin imagen. Al repo anterior le pasó con `front`.

    Las de `perception.rig:` —las dos diagonales— NO pasan por ese vocabulario, y por eso
    viven en otro bloque: no suben ninguna foto, sólo alimentan la fusión de profundidad
    de `planner/heightmap.py`. La cenital es la MISMA `top` de arriba, compartida a
    propósito: percepción y plataforma quieren el mismo punto de vista, y dos cenitales
    que se van separando con los años acaban en una foto y un mapa que no se
    corresponden.
    """
    from src.cell.render import VIEWS

    out: list[str] = []
    for name, spec in cfg["cameras"].items():
        if name not in VIEWS:
            raise ValueError(
                f"cámara {name!r} fuera del vocabulario {VIEWS}. Con otro nombre la foto "
                f"sube a Storage y la base rechaza la fila con un 23514."
            )
        out.append(_camera_xml(name, spec))
    for name, spec in cfg.get("perception", {}).get("rig", {}).items():
        if name in cfg["cameras"]:
            raise ValueError(
                f"la cámara de percepción {name!r} pisa a una de `cameras:`. Si lo que "
                f"quieres es compartirla, ponla sólo en `perception.cameras`."
            )
        out.append(_camera_xml(name, spec))
    return "\n".join(out)


def lighting_for(cfg: dict, level: Level) -> dict:
    """El bloque de luz de este decorado. Los dos declaran las mismas claves `headlight_*`."""
    return cfg["plant"]["lighting"] if level.decor == "plant" else cfg["lighting"]


def _lighting_xml(cfg: dict, level: Level) -> str:
    """Iluminación de esta escena.

    La cenital ve caras horizontales y satura; el alzado ve verticales y sale a oscuras.
    El `fill` diagonal es lo que arregla el segundo sin romper el primero. Las medias RGB
    medidas están en la cabecera de `configs/pallet.yaml`.

    Con `decor: plant` el rig es otro —y con sus propias medias medidas— porque la nave
    tiene luminarias y ventanas y la celda desnuda no. Lo escribe `src/cell/plant.py`.
    """
    if level.decor == "plant":
        return plant.lighting_xml(cfg)
    light = cfg["lighting"]
    overhead = " ".join(str(value) for value in light["overhead_diffuse"])
    spot = " ".join(str(value) for value in light["spot_diffuse"])
    fill_dir = " ".join(str(value) for value in light["fill"]["direction"])
    fill_diffuse = " ".join(str(value) for value in light["fill"]["diffuse"])
    overhead_pos = " ".join(str(value) for value in light["overhead_pos"])
    spot_pos = " ".join(str(value) for value in light["spot_pos"])
    return f"""
    <light pos="{overhead_pos}" dir="0 0 -1" directional="true" diffuse="{overhead}"/>
    <light pos="{spot_pos}" dir="0 0 -1" diffuse="{spot}"/>
    <light pos="0 0 3" dir="{fill_dir}" directional="true" diffuse="{fill_diffuse}"/>"""


def build_mjcf(
    cfg: dict,
    level: Level,
    boxes: list[Box],
    held: HeldPackage | None = None,
    *,
    simplified: bool = False,
    with_robot: bool = True,
) -> str:
    """El MJCF entero, o su variante física sin UR10e para entrenar pesos."""
    if held is not None and not with_robot:
        raise ValueError("una escena sin robot no puede llevar un paquete agarrado")
    held_index = held.index if held is not None else -1
    packages = "\n".join(
        _package_xml(box, held, simplified) for box in boxes if box.index != held_index
    )
    # El paquete que cuelga de `tool` ya no es un cuerpo libre, así que la weld que lo
    # sujetaría no tiene nada que restringir y se deja fuera.
    carried = "".join(
        _package_xml(box, held, simplified) for box in boxes if box.index == held_index
    )
    welds = (
        "\n      ".join(_weld_xml(box) for box in boxes if box.index != held_index)
        if with_robot else ""
    )

    # Mesa y cinta cuelgan de un carril que sale de la caja negra; el camión no, porque
    # su carga ya viene apilada en el remolque. Ver `lane_for`.
    lane = lane_for(cfg, level.source)
    source_fixture = {
        "table": lambda: add_table(cfg["table"]) + "\n" + _belt_xml(cfg, simplified, lane),
        "conveyor": lambda: _belt_xml(cfg, simplified, lane),
        "truck": lambda: _truck_xml(cfg, simplified),
    }[level.source]()
    if lane is not None:
        source_fixture += "\n" + _black_box_xml(cfg, boxes, lane)
    auxiliary_table = add_table(cfg["auxiliary_table"], name="auxiliary_table")

    meshes = (
        ""
        if simplified
        else "\n    ".join(f'<mesh name="{name}" file="{name}.obj"/>' for name in UR10E_MESHES)
    )
    ped = cfg["pedestal"]
    light = lighting_for(cfg, level)
    materials = _MATERIALS + (plant.materials_xml() if level.decor == "plant" else "")
    episode = cfg["episode"]
    robot = _ur10e_xml(cfg, _cups_xml(cfg, simplified), carried, simplified) if with_robot else ""
    actuators = "" if not with_robot else """
    <general class="size4" name="shoulder_pan" joint="shoulder_pan_joint"/>
    <general class="size4" name="shoulder_lift" joint="shoulder_lift_joint"/>
    <general class="size3_limited" name="elbow" joint="elbow_joint"/>
    <general class="size2" name="wrist_1" joint="wrist_1_joint"/>
    <general class="size2" name="wrist_2" joint="wrist_2_joint"/>
    <general class="size2" name="wrist_3" joint="wrist_3_joint"/>"""
    sensors = "" if not with_robot else """
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>"""

    return f"""<mujoco model="celda de paletizado UR10e · {level.name}">
  <compiler angle="radian" autolimits="true" meshdir="{mesh_dir(cfg)}"/>
  <option timestep="{episode['timestep']}" gravity="0 0 -9.81"
          integrator="implicitfast" cone="elliptic"/>
  <size njmax="4000" nconmax="800"/>
  <default>
    <default class="ur10e">
      <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
      <!-- Clase de colisión 2: el brazo choca con la celda pero NO con el paquete que
           cuelga rígidamente de sus propias ventosas. -->
      <geom friction="0.8 0.01 0.001" contype="{ARM_CONTACT}" conaffinity="{ARM_CONTACT}"/>
      <general biastype="affine" ctrlrange="-6.2831 6.2831"
               gainprm="9000" biasprm="0 -9000 -700"/>
      <default class="size4"><joint damping="10"/><general forcerange="-330 330"/></default>
      <default class="size3_limited">
        <joint damping="5" range="-3.1415 3.1415"/>
        <general forcerange="-150 150" ctrlrange="-3.1415 3.1415"/>
      </default>
      <default class="size2"><joint damping="2"/><general forcerange="-56 56"/></default>
      <default class="robot_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
    </default>
  </default>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <headlight ambient="{light['headlight_ambient']} {light['headlight_ambient']} {light['headlight_ambient']}"
               diffuse="{light['headlight_diffuse']} {light['headlight_diffuse']} {light['headlight_diffuse']}"
               specular="{light['headlight_specular']} {light['headlight_specular']} {light['headlight_specular']}"/>
  </visual>
  <asset>{materials}
    {meshes}
  </asset>
  <worldbody>{_lighting_xml(cfg, level)}
{_cameras_xml(cfg)}
{_floor_xml(cfg, level)}
    <geom name="pedestal" type="cylinder" pos="{ped['center'][0]} {ped['center'][1]} {ped['height'] / 2}"
          size="{ped['radius']} {ped['height'] / 2}" rgba="0.22 0.25 0.28 1" conaffinity="{SOLID}"/>
{source_fixture}
{auxiliary_table}
{_decor_xml(cfg, level, simplified)}
{_stability_beam_xml(cfg)}
{_pallet_xml(cfg, simplified)}
{robot}
{packages}
  </worldbody>
  <equality>
      <weld name="pallet_anchor" body1="pallet" solref="0.002 1"
            solimp="0.95 0.99 0.001 0.5 2"/>
      {welds}
  </equality>
  <contact>
    <exclude body1="world" body2="pallet"/>
  </contact>
  <actuator>
    {actuators}
  </actuator>
  <sensor>
    {sensors}
  </sensor>
</mujoco>"""


# ─────────────────────────────────────────────────────────────────────────────
# La escena viva
# ─────────────────────────────────────────────────────────────────────────────

class PalletScene:
    """El modelo compilado, sus datos y todo lo que hay que saber para trabajarlo.

    Es lo que viaja a `detector.observe(scene)`, `gauge.measure(scene, ...)`,
    `heightmap.measure(scene)` y `supply.present(scene)`. Ninguno de ellos puede mirar
    `boxes`: ahí está la VERDAD, y leerla es exactamente lo que hacen los stubs-oráculo,
    que viven en ficheros con el nombre puesto.
    """

    def __init__(self, cfg: dict, level: Level, boxes: list[Box], seed: int,
                 *, simplified: bool = False, with_robot: bool = True):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        self.level = level
        self.boxes = boxes
        self.seed = seed
        self.simplified = simplified
        self.with_robot = with_robot
        self.held: HeldPackage | None = None
        self.rng = np.random.default_rng(seed)
        # La rejilla de espera dentro de la caja negra, la MISMA que dimensionó el
        # cerramiento al construir el MJCF. Se calcula una vez: si `park_box` la
        # recalculara por su cuenta y los dos no coincidieran, los que esperan
        # aparecerían atravesando la pared y no se vería por qué.
        lane = lane_for(cfg, level.source)
        self._parking = None if lane is None else parking_grid(cfg, boxes, lane)
        # Tiempo SIMULADO desde que arrancó el episodio. Es el que va a `events.ts`, no
        # el de reloj: así dos ejecuciones de la misma semilla dan la misma línea
        # temporal por rápida que sea la máquina.
        self.clock = 0.0
        self.viewer = None
        # Segundos simulados por segundo de reloj de pared, SÓLO con visor. 0 = a lo que
        # dé la máquina, que es lo que quiere el headless. Lo pone `run_episode`.
        self.speed = 0.0
        self._wall_origin = 0.0
        self._next_frame = 0.0

        self._compile(build_mjcf(
            cfg, level, boxes, None, simplified=simplified, with_robot=with_robot,
        ))
        self.home = np.asarray(cfg["robot"]["ik_seed_qpos"], dtype=float)
        self.observe_qpos = np.asarray(cfg["robot"]["observe_qpos"], dtype=float)
        if with_robot:
            self.data.qpos[self.arm_qpos] = self.home
            self.data.ctrl[:] = self.home
        mujoco.mj_forward(self.model, self.data)

    # ── el modelo ────────────────────────────────────────────────────────────

    def _compile(self, mjcf: str) -> None:
        """Compila y cachea los índices. Aparte de `__init__` porque sellar una ventosa
        RECONSTRUYE la celda: el paquete deja de ser cuerpo libre y pasa a ser parte de
        la herramienta, y eso cambia el modelo."""
        mujoco = self.mujoco
        self.mjcf = mjcf
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

        self.arm_qpos: list[int] = []
        self.arm_dofs: list[int] = []
        self.tcp_site = -1
        self.ft_site = -1
        self.tool_body = -1
        if self.with_robot:
            names = self.cfg["robot"]["joints"]
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in names
            ]
            self.arm_qpos = [self.model.jnt_qposadr[index] for index in joint_ids]
            self.arm_dofs = [self.model.jnt_dofadr[index] for index in joint_ids]
            self.tcp_site = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE
            )
            self.ft_site = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "ft_site"
            )
            self.tool_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "tool"
            )
        self.pallet_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pallet")
        self.pallet_weld = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "pallet_anchor"
        )
        pallet_joints = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("pallet_x", "pallet_y", "pallet_z", "pallet_rx", "pallet_ry")
        ]
        self.pallet_qpos = [self.model.jnt_qposadr[index] for index in pallet_joints]
        self.pallet_dofs = [self.model.jnt_dofadr[index] for index in pallet_joints]
        self.stability_beam_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "stability_beam"
        )
        self.stability_beam_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "stability_beam"
        )
        self.stability_beam_mocap = int(
            self.model.body_mocapid[self.stability_beam_body]
        )

    def rebuild(self, held: HeldPackage | None) -> None:
        """Vuelve a compilar con el paquete colgando de la herramienta, o sin él.

        Se conserva el estado: la pose del brazo, la de cada caja libre y el reloj. Lo
        único que cambia es de quién cuelga el cartón sellado.
        """
        if not self.with_robot:
            raise RuntimeError("una escena sin robot no se puede reconstruir para agarrar")
        mujoco = self.mujoco
        arm = np.asarray(self.data.qpos[self.arm_qpos]).copy()
        pallet_qpos = np.asarray(self.data.qpos[self.pallet_qpos]).copy()
        pallet_qvel = np.asarray(self.data.qvel[self.pallet_dofs]).copy()
        mocap_pos, mocap_quat = self.data.mocap_pos.copy(), self.data.mocap_quat.copy()
        poses = {box.index: self.box_pose(box.index) for box in self.boxes}
        pallet_locked = (
            bool(self.data.eq_active[self.pallet_weld]) if self.pallet_weld >= 0 else True
        )

        self.held = held
        self._compile(build_mjcf(
            self.cfg, self.level, self.boxes, held,
            simplified=self.simplified, with_robot=True,
        ))

        self.data.qpos[self.arm_qpos] = arm
        self.data.ctrl[:] = arm
        self.data.qpos[self.pallet_qpos] = pallet_qpos
        self.data.qvel[self.pallet_dofs] = pallet_qvel
        self.data.mocap_pos[:] = mocap_pos
        self.data.mocap_quat[:] = mocap_quat
        if self.pallet_weld >= 0:
            self.data.eq_active[self.pallet_weld] = pallet_locked
        for box in self.boxes:
            if held is not None and box.index == held.index:
                continue
            position, quaternion = poses[box.index]
            self._write_free_pose(box.index, position, quaternion)
        mujoco.mj_forward(self.model, self.data)
        self._reload_viewer()

    # ── estado de las cajas ──────────────────────────────────────────────────

    def body_id(self, index: int) -> int:
        return self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.boxes[index].body
        )

    def box_pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Dónde está de verdad una caja: posición y cuaternión del mundo."""
        body = self.body_id(index)
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def box_yaw(self, index: int) -> float:
        """El giro con el que la caja realmente yace, en radianes.

        Hace de lo que devolvería una estimación de pose RGB-D. Una columna apilada a
        mano nunca está a escuadra con el remolque, y la herramienta tiene que alinearse
        con el cartón sobre el que va a sellar, no con la bahía.
        """
        _, (qw, qx, qy, qz) = self.box_pose(index)
        return float(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def box_top_center(self, index: int) -> np.ndarray:
        """El centro de la cara superior: donde la herramienta tiene que sellar."""
        position, _ = self.box_pose(index)
        return position + np.array([0.0, 0.0, self.boxes[index].dims_m[2] / 2])

    def _write_free_pose(self, index: int, position, quaternion) -> None:
        joint = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, self.boxes[index].joint
        )
        if joint < 0:
            return                      # está colgando de la herramienta: no tiene joint
        address = self.model.jnt_qposadr[joint]
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = quaternion
        dof = self.model.jnt_dofadr[joint]
        self.data.qvel[dof : dof + 6] = 0.0

    def place_box(self, index: int, position, yaw: float = 0.0) -> None:
        """Pone una caja donde se le diga, quieta. Sólo para montar la escena inicial.

        Esto NO es una maniobra: es cómo la fuente coloca su carga antes de que empiece
        el episodio —los bultos en la mesa, la cola de la cinta, las columnas del
        remolque—. Durante el episodio nada se teletransporta.
        """
        half = math.radians(0.0) if yaw is None else yaw / 2
        quaternion = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        self._write_free_pose(index, np.asarray(position, dtype=float), quaternion)
        self.mujoco.mj_forward(self.model, self.data)

    def park_box(self, index: int) -> None:
        """Deja una caja esperando DENTRO de la caja negra, en su hueco de la rejilla.

        Antes era una fila que se iba de la escena —21 m con treinta bultos—. Ahora los
        que esperan están donde dice la ficción: fuera de la vista, no en el infinito.
        Sin carril —el camión— se conserva la fila de siempre, que allí no se ve porque
        la carga entera está en el remolque desde el principio.
        """
        height = self.boxes[index].dims_m[2]
        if self._parking is None:
            self.place_box(index, (-3.0 - index * 0.7, 0.0, height / 2 + 0.002))
            return
        x, y = self._parking[index]
        self.place_box(index, (x, y, height / 2 + 0.002))

    # ── física ───────────────────────────────────────────────────────────────

    @property
    def steps_per_tick(self) -> int:
        return max(1, round(1.0 / (self.cfg["episode"]["control_hz"] * self.model.opt.timestep)))

    def step(self, seconds: float) -> None:
        """Avanza la física y el reloj SIMULADO."""
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self.mujoco.mj_step(self.model, self.data)
            self.clock += self.model.opt.timestep
            self.sync_viewer()

    def settle(self, seconds: float | None = None) -> None:
        """Deja que la escena se asiente. Medir antes de esto es medir cajas cayendo."""
        self.step(self.cfg["motion"]["settle_seconds"] if seconds is None else seconds)

    def settle_until_rest(self, max_seconds: float = 4.0,
                          threshold: float = 0.002) -> float:
        """Asienta hasta que los bultos están QUIETOS, no durante un rato fijo.

        Un tiempo fijo es una apuesta sobre cuánto tarda la escena en calmarse, y la
        pierde en cuanto el nivel cambia: con el asentado de 0.4 s, el nivel 12 arrancaba
        con 0.019 m/s de velocidad residual y el 13 con 3.89 m/s. Eso no es ruido, es que
        las cajas siguen moviéndose mientras el brazo viaja —un par de segundos— y cuando
        llega, la que vio la percepción ya no está donde estaba. Salía como si la ventosa
        resbalase.

        Devuelve los segundos simulados que hizo falta, para poder anotarlos.
        """
        step = 0.1
        waited = 0.0
        while waited < max_seconds:
            self.step(step)
            waited += step
            if self.max_box_speed() < threshold:
                break
        return waited

    def max_box_speed(self) -> float:
        """La velocidad lineal del bulto que más se mueve, en m/s."""
        fastest = 0.0
        for box in self.boxes:
            start = self.model.body_dofadr[self.body_id(box.index)]
            if start < 0:
                continue
            speed = float(np.linalg.norm(self.data.qvel[start:start + 3]))
            fastest = max(fastest, speed)
        return fastest

    def _reload_viewer(self) -> None:
        """Mantiene la misma ventana GLFW cuando el árbol cinemático cambia al sellar."""
        if self.viewer is None:
            return
        simulate = getattr(self.viewer, "_get_sim", lambda: None)()
        if simulate is None:
            return
        simulate.load(self.model, self.data, "")

    def sync_viewer(self) -> None:
        """Refresca el visor, y es también donde el episodio lleva el ritmo.

        Los tres bucles de física —`step`, `move_joints` y `step_physics`— pasan por
        aquí una vez por PASO de 2 ms, así que éste es el único sitio donde hace falta
        poner las dos cosas, y por eso ningún otro cambia.

        **Acompasar.** No había nada que atase el reloj simulado al de pared: `--speed`
        sólo decidía si el brazo teletransporta (`speed <= 0`) y cualquier valor positivo
        corría igual. Aquí `speed` pasa a ser lo que dice ser, segundos simulados por
        segundo real, y `--speed 4` se ve cuatro veces más rápido.

        **Y no sincronizar 500 veces por segundo.** Un episodio del nivel 21 son ~46.000
        pasos: sincronizar en cada uno es ocho veces la tasa de pantalla, y lo que se ve
        no es más fluido, es más lento. Se limita a 60 Hz de reloj de pared.

        El reancle es por `solve_ik`: sus 600 `mj_forward` no pasan por aquí, así que
        tras una convergencia lenta el reloj simulado se queda atrás y sin reanclar el
        visor correría a saltos recuperando el retraso.
        """
        if self.viewer is None or not self.viewer.is_running():
            return

        now = time.monotonic()
        if self.speed > 0:
            if self._wall_origin == 0.0:
                self._wall_origin = now
            delay = self._wall_origin + self.clock / self.speed - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            elif delay < -0.5:
                self._wall_origin = now - self.clock / self.speed
        if now < self._next_frame:
            return
        self._next_frame = now + FRAME_SECONDS

        # `m` es también el atajo de MuJoCo para "Center of Mass". Ver el porqué en
        # `render.silence_com_markers`. El import va aquí y no arriba porque
        # `render` importa de este módulo.
        from src.cell.render import draw_overlay, silence_com_markers

        silence_com_markers(self.viewer, self.mujoco)
        draw_overlay(self)
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ── cotas de la celda, en un solo sitio ──────────────────────────────────

    @property
    def deck_z(self) -> float:
        """La cara superior del palé: sobre ella se apila."""
        return float(self.cfg["pallet"]["deck_thickness"])

    @property
    def pallet_center(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.cfg["pallet"]["center"])

    @property
    def pallet_dims(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.cfg["pallet"]["dims"])

    @property
    def station(self) -> tuple[float, float]:
        """Dónde el paquete se PARA para que el brazo lo coja."""
        if self.level.source == "truck":
            bay = self.cfg["truck"]
            return (bay["origin"][0] + bay["width"] / 2, bay["origin"][1] + bay["depth"] / 2)
        if self.level.source == "conveyor":
            return tuple(float(value) for value in self.cfg["conveyor"]["station"])
        return tuple(float(value) for value in self.cfg["table"]["center"])

    @property
    def surface_z(self) -> float:
        """La cota sobre la que descansa la carga antes de cogerla."""
        return {
            "table": lambda: float(self.cfg["table"]["height"]),
            "conveyor": lambda: float(self.cfg["conveyor"]["height"]),
            "truck": lambda: float(self.cfg["truck"]["floor_height"]),
        }[self.level.source]()


def build_scene(cfg: dict | None = None, level_id: int | None = None, seed: int = 0,
                *, simplified: bool = False, with_robot: bool = True) -> PalletScene:
    """La celda montada y con la física asentada, para el nivel que se pida.

    Las cajas salen aparcadas fuera de escena: ponerlas donde empiezan es trabajo de la
    fuente (`src/cell/conveyor.py::make_supply(...).stage(scene)`), porque dónde empieza
    la carga es precisamente lo que distingue una tarea de otra.

    ``with_robot=False`` conserva palé, cajas y ensayo de estabilidad, pero elimina
    cuerpo, actuadores, sensores y welds del UR10e. Sólo lo usa el banco de pesos: un
    episodio de celda siempre deja el valor por defecto.
    """
    cfg = load_configs() if cfg is None else cfg
    level = level_for(cfg, level_id)
    boxes = build_catalogue(cfg, level, seed)
    scene = PalletScene(
        cfg, level, boxes, seed, simplified=simplified, with_robot=with_robot,
    )
    scene.settle(0.20)
    return scene


def box_pose(scene: PalletScene, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Dónde está de verdad una caja. Azúcar sobre `PalletScene.box_pose`."""
    return scene.box_pose(index)
