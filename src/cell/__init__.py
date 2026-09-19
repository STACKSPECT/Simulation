"""
La celda: el banco de trabajo. Mesa, frame del TCP y orientación de cámaras.

Portado desde el demostrador (`tools/stable_pallet/simulator.py`), que es de donde sale
toda la capa de ejecución de este repo.

Lo que trae y se usa aquí sin cambios:

    TCP_SITE                       nombre del site que la IK usa como frame del TCP
    TOOL_DOWN                      la orientación de trabajo de la herramienta
    add_table(cfg)                 la mesa, a partir de configs/scene.yaml
    tcp_frame(position, yaw)       pose de agarre desde un punto y el giro de la caja
    lookat_quat(position, lookat)  cuaternión de una cámara que mira a un punto

`tcp_frame` es lo que hace que todas las poses compartan orientación, y de ahí sale que
el trayecto en línea recta funcione. No lo reescribas "mejor".

── UN CAMBIO RESPECTO A LO QUE DECÍA LA CABECERA ────────────────────────────────

Era `tcp_frame(position, closing)`: un punto y una **dirección de cierre**, que es lo que
necesita una pinza paralela para decidir por qué dos caras agarra. Aquí la herramienta es
un OnRobot VGP20 de 16 ventosas y no hay cierre: se sella por arriba, siempre por arriba,
y lo único que queda por decidir es el giro con el que la rejilla de ventosas se alinea
con la caja. Así que la firma pasa a ser `tcp_frame(position, yaw)`.

No es una simplificación gratuita: es la diferencia que quita el techo de ocupación del
palé. Con pinza, la holgura entre cajas vecinas la imponía el grueso del dedo y el palé
no pasaba del 72-76 %; con ventosa la holgura la impone el error de seguimiento del
descenso, y es una decisión de control, no de mecánica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# El site que `src/cell/scene.py` cuelga de la herramienta, en la cara de las ventosas.
# `configs/scene.yaml: robot.ik_frame` tiene que decir esto mismo.
TCP_SITE = "tcp"

# La orientación de trabajo: la herramienta mira al suelo. Es `diag(1, -1, -1)` y no la
# identidad porque el frame de la herramienta apunta hacia abajo, de modo que su eje Y
# corre CONTRA el del mundo.
#
# Esto no es trivia: el bloque de ventosas que agarra una caja estrecha queda
# descentrado respecto al cuerpo, y el brazo tiene que cancelar ese desplazamiento en
# cada movimiento. Cancelarlo directamente en coordenadas de mundo lo DUPLICA en ese eje
# en vez de quitarlo, y entonces la fila exterior de ventosas cuelga por fuera del
# cartón. Costó una tanda entera de agarres por la esquina.
TOOL_DOWN = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Pose:
    """Dónde tiene que ir el TCP y con qué orientación. Metros y radianes."""

    position: np.ndarray            # [x, y, z] en el mundo
    rotation: np.ndarray            # 3x3, filas del frame del TCP en el mundo

    @property
    def yaw(self) -> float:
        """El giro alrededor de Z que separa esta pose de `TOOL_DOWN`."""
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

    def lifted(self, z: float) -> Pose:
        """La misma pose a otra cota. Es el 90 % de los waypoints de un ciclo."""
        return Pose(np.array([self.position[0], self.position[1], z]), self.rotation)


def rotation_z(angle: float) -> np.ndarray:
    """Giro alrededor del eje Z del mundo, en radianes."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def tcp_frame(position, yaw: float) -> Pose:
    """Pose de agarre sobre un punto, con la rejilla de ventosas girada `yaw` radianes.

    Todas las poses del episodio salen de aquí, y por eso todas comparten orientación:
    es lo que permite que el trayecto entre dos cualesquiera sea una recta en cartesiano
    sin que la muñeca tenga que recolocarse por el camino.
    """
    return Pose(np.asarray(position, dtype=float), rotation_z(yaw) @ TOOL_DOWN)


def lookat_quat(position, lookat, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Cuaternión `(w, x, y, z)` de una cámara puesta en `position` que mira a `lookat`.

    Las cámaras de MuJoCo miran a lo largo de su **-Z** con su **+Y** arriba, que es
    justo al revés de lo que uno escribe. Declararlas con `position`/`lookat` en el YAML
    y convertir aquí evita tener que razonar `xyaxes` a mano cada vez que se mueve una.

    `up` importa más de lo que parece en la cenital: sin él, la vista de arriba sale
    girada 90 grados respecto a como el alzado dibuja el palé, y las cotas de la interfaz
    dejan de casar con la foto. Por eso `configs/pallet.yaml` se lo pasa explícitamente.
    """
    position = np.asarray(position, dtype=float)
    lookat = np.asarray(lookat, dtype=float)
    up = np.asarray(up, dtype=float)

    backward = position - lookat
    norm = np.linalg.norm(backward)
    if norm < 1e-9:
        raise ValueError("una cámara no puede estar en el punto que mira")
    z_axis = backward / norm

    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # `up` es paralelo a la dirección de vista: cualquier horizontal vale.
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return mat_to_quat(np.column_stack((x_axis, y_axis, z_axis)))


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3x3 -> cuaternión `(w, x, y, z)`, que es el orden que quiere MuJoCo."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
        scale = np.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
        w = (matrix[k, j] - matrix[j, k]) / scale
        components = [0.0, 0.0, 0.0]
        components[i] = 0.25 * scale
        components[j] = (matrix[j, i] + matrix[i, j]) / scale
        components[k] = (matrix[k, i] + matrix[i, k]) / scale
        x, y, z = components
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def add_table(cfg: dict) -> str:
    """La mesa de trabajo, a partir del bloque `table` de `configs/scene.yaml`.

    `size` son SEMIEJES porque es lo que quiere MuJoCo, y la clave se llama así por eso:
    en el mismo YAML `dims` son dimensiones completas. Un factor 2 escondido en un fichero
    de configuración no lo caza nadie.

    `conaffinity="3"` la hace sólida tanto para los paquetes (clase 1) como para el brazo
    (clase 2): una caja que se cae de la mesa se queda encima, y el brazo choca con ella
    en vez de atravesarla.
    """
    x, y = cfg["center"]
    half_x, half_y, half_z = cfg["size"]
    friction = " ".join(str(value) for value in cfg["friction"])
    rgba = " ".join(str(value) for value in cfg["rgba"])
    top = cfg["height"]
    legs = "\n".join(
        f'    <geom type="box" pos="{x + dx:.4f} {y + dy:.4f} {top / 2 - 0.04:.4f}" '
        f'size="0.035 0.035 {top / 2 - 0.04:.4f}" material="frame" '
        'contype="0" conaffinity="0"/>'
        for dx in (-half_x + 0.05, half_x - 0.05)
        for dy in (-half_y + 0.05, half_y - 0.05)
    )
    return f"""
    <geom name="table" type="box" pos="{x} {y} {top - half_z:.4f}"
          size="{half_x} {half_y} {half_z}" rgba="{rgba}"
          friction="{friction}" conaffinity="3"/>
{legs}"""
