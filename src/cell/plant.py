"""La nave que rodea la celda en los niveles con `decor: plant`.

`assets/planta_industrial_v2.xml` es una exportación de Blender: 152 líneas de MJCF
plano —cajas y cilindros con `rgba` y nada más— con paredes, ventanas, luminarias,
estantería, depósitos, banco de trabajo, bidones y señalización. De ese fichero se usa
**sólo el `<worldbody>`**, y ni siquiera entero.

── POR QUÉ SE TRANSFORMA Y NO SE INCLUYE ─────────────────────────────────────
Cuatro cosas de la exportación chocan con esta celda y las cuatro fallan tarde:

  `<light>` ×4      La nave apunta sus tres LED a y = 1,63 —el fondo del taller—, que
                    es donde estaban las máquinas. La zona de trabajo se quedaba a
                    oscuras. El rig de este decorado lo escribe `lighting_xml`, abajo.
  `<camera>`        Se llama `overview`, y `scene._cameras_xml` valida los nombres
                    contra `render.VIEWS`: con otro nombre la foto sube a Storage y
                    LUEGO la base rechaza la fila con un 23514.
  `geom "floor"`    La celda ya tiene su plano, con la fricción calibrada.
  `<freejoint>`     Las tres cajas de la estantería son cuerpos libres. Meterían 21
                    qpos ajenos al episodio y tres cuerpos más que asentar cada vez
                    que se reconstruye la escena. Se aplanan a geom estático: se
                    siguen viendo en su balda y no pesan nada.

Y todo lo que sí se emite sale **sin contacto**. Es un fondo: el brazo tiene 1,3 m de
alcance desde (0,60, −0,72) y el geom de la nave más cercano queda a más de 2 m, así
que los contactos serían coste puro y una manera de que el codo se enganche en un
depósito. Mismo criterio que `scene._industrial_xml`, que ya es decorado.

── DÓNDE SE PLANTA ───────────────────────────────────────────────────────────
La nave trae marcado en el suelo un rectángulo libre de 4,8 × 3 m —x ∈ [−2,4, 2,4],
y ∈ [−2,2, 0,8] en SUS coordenadas— y la celda ocupa x ∈ [−0,62, +1,75],
y ∈ [−1,28, +0,40] en las del mundo (palé, bahía del remolque, mesa auxiliar y
pedestal). `plant.offset` en `configs/pallet.yaml` es lo que hace coincidir los dos
centros; el número está medido ahí, no elegido.

Nota de procedencia: el original vive en `~/Descargas/planta_industrial/` junto al
`.blend` que lo genera, y `export_mujoco.py` lo sobrescribe en cada re-exportación.
Por eso aquí hay una COPIA. Si se vuelve a exportar, se vuelve a copiar —y
`tests/test_pallet.py` avisa si la paleta de colores cambió.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

# Los trece colores planos de la exportación, y el material con el que se pintan aquí.
# La nave no trae `<asset>`: todo es `rgba` sin especular ni rugosidad, y de mapear eso
# a materiales de verdad sale la mitad de la ambientación —hormigón mate contra acero
# especular contra depósitos satinados—. La clave es el `rgba` literal del fichero.
MATERIAL_FOR_RGBA = {
    "0.47 0.59 0.63 1": "plant_steel",
    "0.045 0.075 0.09 1": "plant_dark",
    "1 0.61 0.045 1": "plant_yellow",
    "0.025 0.33 0.36 1": "plant_teal",
    "0.85 0.92 0.9 1": "plant_white",
    "0.055 0.17 0.19 1": "plant_wall_base",
    "0.78 0.9 1 1": "plant_diffuser",
    "0.64 0.035 0.025 1": "plant_red",
    "0.18 0.42 0.57 1": "plant_window",
    "0.49 0.55 0.55 1": "plant_wall",
    "0.32 0.38 0.4 1": "plant_concrete",
    "0.14 0.76 0.3 1": "plant_pilot",
    # El cartón de la estantería es el mismo marrón que el palé: se reutiliza su
    # material en vez de declarar un decimocuarto que sólo se distinguiría con lupa.
    "0.65 0.39 0.18 1": "pallet_wood",
}

# Los dos emisivos son lo que hace que la nave se vea ENCENDIDA y no sólo iluminada: sin
# ellos los difusores de las luminarias salen del mismo gris que su carcasa y las
# ventanas —que son el origen de la rasante cálida— se leen como paneles opacos.
_PLANT_MATERIALS = """
    <material name="plant_concrete" rgba="0.32 0.38 0.40 1" roughness="0.95" reflectance="0.02"/>
    <material name="plant_wall" rgba="0.49 0.55 0.55 1" roughness="0.88"/>
    <material name="plant_wall_base" rgba="0.055 0.17 0.19 1" roughness="0.72"/>
    <material name="plant_dark" rgba="0.045 0.075 0.09 1" roughness="0.55"/>
    <material name="plant_steel" rgba="0.47 0.59 0.63 1" metallic="0.55" roughness="0.30"/>
    <material name="plant_teal" rgba="0.025 0.33 0.36 1" metallic="0.25" roughness="0.35"/>
    <material name="plant_yellow" rgba="1 0.61 0.045 1" roughness="0.55"/>
    <material name="plant_white" rgba="0.85 0.92 0.90 1" roughness="0.60"/>
    <material name="plant_red" rgba="0.64 0.035 0.025 1" roughness="0.45"/>
    <material name="plant_diffuser" rgba="0.90 0.94 1 1" emission="0.85"/>
    <material name="plant_window" rgba="0.95 0.66 0.38 1" emission="0.55"/>
    <material name="plant_pilot" rgba="0.14 0.76 0.30 1" emission="0.70"/>"""

# Del fichero original, lo que NO se emite. Ver la cabecera.
_SKIPPED_GEOMS = frozenset({"floor"})

# ── EL PÓRTICO ────────────────────────────────────────────────────────────────
# La nave trae sus tres luminarias colgadas del fondo (y = 1,63 en sus coordenadas,
# y = 1,89 en las del mundo), que es donde estaban los depósitos y el banco. A la altura
# del cono —45 grados desde z = 3,055 con exponente 10— eso deja la bahía del remolque,
# en y ∈ [−1,28, −0,40], prácticamente sin luz propia.
#
# Así que la nave gana una línea de luz sobre la celda, que es lo que tendría una nave a
# la que le meten una celda robotizada: una viga cruzada de pared a pared y tres
# luminarias colgando. La viga copia cota y sección de las dos que ya trae la nave
# (`wall_back_beam`, `wall_left_beam`: z = 3,45, semiejes 3,9 × 0,12 × 0,06) y las
# luminarias copian las suyas, para que no se note el injerto.
#
# `GANTRY_Y` es el centro en Y de la celda —(−1,28 + 0,40) / 2— y `LAMP_X` reparte tres
# luminarias de 1,4 m de largo cada 1,85 m alrededor de su centro en X: 0,45 m de hueco
# entre carcasas, que a esta altura se lee como una línea continua.
GANTRY_Y = -0.44
GANTRY_Z = 3.45
LAMP_X = (-1.28, 0.57, 2.42)
LAMP_Z = 3.06


def materials_xml() -> str:
    """Los materiales de la paleta de la nave, para el `<asset>` de la escena."""
    return _PLANT_MATERIALS


def decor_xml(cfg: dict, simplified: bool) -> str:
    """La nave entera: el atrezo de la exportación más el pórtico de la celda."""
    if simplified:
        return ""
    plant = cfg["plant"]
    offset_x, offset_y = (float(value) for value in plant["offset"])
    mjcf = Path(cfg["_root"]) / plant["mjcf"]
    return f"{_scenery_xml(mjcf, offset_x, offset_y)}\n{_gantry_xml()}"


def lighting_xml(cfg: dict) -> str:
    """El rig de la nave: tres luminarias frías y una rasante cálida de ventana.

    Las tres ventanas están en el muro −X (x = −3,85 en coordenadas de la nave,
    z = 2,55), así que la rasante entra desde ahí y barre la celda hacia +X. Eso deja a
    oscuras justo las caras que mira la cámara `side`, en (0,60, −2,30, 0,80) — y por eso
    el relleno diagonal de la celda se queda tal cual. Es el mismo compromiso que
    documenta la cabecera de `lighting:` y la razón de que ese relleno exista.

    Cinco luces en total. El límite de render de MuJoCo son ocho.

    ── LAS DOS DIRECCIONALES NO PROYECTAN SOMBRA, Y ES A PROPÓSITO ────────────
    El mapa de sombras de MuJoCo tiene una sola resolución (`<quality shadowsize>`) y
    hay que repartirla por toda la extensión de la escena. Con la celda desnuda esa
    extensión es de unos 4 m; con la nave alrededor pasa a 14, y una direccional —que
    cubre el escenario entero— se queda con menos de una décima parte de los téxeles
    que tenía. El resultado es un escalonado muy visible sobre las paredes, los
    depósitos y el armario, justo el fondo que este decorado existe para enseñar.
    Apagarlas deja la sombra a las tres luminarias, que son locales y sí tienen
    resolución de sobra, y le ahorra a la escena dos pasadas de sombra por fotograma.
    Ni el sol ni el rebote del techo son sombras que se echen de menos.
    """
    light = cfg["plant"]["lighting"]
    leds = _triplet(light["leds"]["diffuse"])
    lamps = "\n".join(
        f'    <light name="plant_led_{index}" pos="{x} {GANTRY_Y} {LAMP_Z}" dir="0 0 -1" '
        f'diffuse="{leds}" specular="0.12 0.12 0.14" cutoff="60" exponent="4"/>'
        for index, x in enumerate(LAMP_X, start=1)
    )
    return f"""
{lamps}
    <light name="plant_window" pos="0 0 3" dir="{_triplet(light['window']['direction'])}"
           directional="true" castshadow="false"
           diffuse="{_triplet(light['window']['diffuse'])}"/>
    <light name="plant_fill" pos="0 0 3" dir="{_triplet(light['fill']['direction'])}"
           directional="true" castshadow="false"
           diffuse="{_triplet(light['fill']['diffuse'])}"/>"""


# ─────────────────────────────────────────────────────────────────────────────


def _triplet(values) -> str:
    return " ".join(str(value) for value in values)


@lru_cache(maxsize=4)
def _scenery_xml(mjcf: Path, offset_x: float, offset_y: float) -> str:
    """El `<worldbody>` de la exportación, desplazado, prefijado y sin contacto.

    En caché porque `scene.build_mjcf` se llama una vez por sellado de la ventosa —ocho
    veces en un episodio de ocho bultos— y reparsear 152 líneas de XML cada vez no
    aporta nada.
    """
    worldbody = ElementTree.parse(mjcf).getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"{mjcf}: la exportación no trae <worldbody>")
    out: list[str] = []
    for node in worldbody:
        if node.tag == "geom":
            attrs = dict(node.attrib)
        elif node.tag == "body":
            attrs = _flattened(node, mjcf)
        else:
            continue          # <light> y <camera>: los pone esta celda. Ver la cabecera.
        if attrs.get("name") in _SKIPPED_GEOMS:
            continue
        out.append(_geom_xml(attrs, offset_x, offset_y, mjcf))
    return "\n".join(out)


def _flattened(body: ElementTree.Element, mjcf: Path) -> dict[str, str]:
    """Un cuerpo libre de la exportación, aplanado a geom estático en su sitio."""
    geom = body.find("geom")
    if geom is None:
        raise ValueError(f"{mjcf}: el cuerpo {body.get('name')!r} no trae geom")
    # El exportador de Blender sube `pos` y `quat` del geom al cuerpo, así que la pose
    # está en el cuerpo y la forma en el geom. `mass` se cae con el cuerpo libre.
    attrs = {key: value for key, value in geom.attrib.items() if key != "mass"}
    attrs["pos"] = body.get("pos", "0 0 0")
    attrs["quat"] = body.get("quat", "1 0 0 0")
    return attrs


def _geom_xml(attrs: dict[str, str], offset_x: float, offset_y: float, mjcf: Path) -> str:
    name = attrs["name"]
    rgba = attrs["rgba"]
    if rgba not in MATERIAL_FOR_RGBA:
        raise ValueError(
            f"{mjcf}: el geom {name!r} usa el color {rgba!r}, que no tiene material en "
            f"MATERIAL_FOR_RGBA. Si se ha re-exportado la nave desde Blender con una "
            f"paleta nueva, hay que declararlo en src/cell/plant.py."
        )
    x, y, z = (float(value) for value in attrs["pos"].split())
    return (
        f'    <geom name="plant_{name}" type="{attrs["type"]}" size="{attrs["size"]}" '
        f'pos="{x + offset_x:.4f} {y + offset_y:.4f} {z}" '
        f'quat="{attrs.get("quat", "1 0 0 0")}" material="{MATERIAL_FOR_RGBA[rgba]}" '
        f'contype="0" conaffinity="0"/>'
    )


def _gantry_xml() -> str:
    """La viga cruzada sobre la celda y sus tres luminarias. Ver el bloque de arriba."""
    lamps = "\n".join(
        f"""    <geom name="plant_gantry_rod_{index}" type="box"
          pos="{x} {GANTRY_Y} {(GANTRY_Z + LAMP_Z) / 2:.4f}"
          size="0.0275 0.0275 {(GANTRY_Z - LAMP_Z) / 2:.4f}"
          material="plant_steel" contype="0" conaffinity="0"/>
    <geom name="plant_gantry_housing_{index}" type="box" pos="{x} {GANTRY_Y} {LAMP_Z + 0.09:.4f}"
          size="0.7 0.17 0.05" material="plant_dark" contype="0" conaffinity="0"/>
    <geom name="plant_gantry_diffuser_{index}" type="box" pos="{x} {GANTRY_Y} {LAMP_Z + 0.025:.4f}"
          size="0.645 0.135 0.015" material="plant_diffuser" contype="0" conaffinity="0"/>"""
        for index, x in enumerate(LAMP_X, start=1)
    )
    return f"""    <geom name="plant_gantry_beam" type="box" pos="0.57 {GANTRY_Y} {GANTRY_Z}"
          size="3.9 0.12 0.06" material="plant_dark" contype="0" conaffinity="0"/>
{lamps}"""
