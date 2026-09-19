"""
La fuente de suministro: de dónde sale el paquete que el brazo tiene que coger.

    present(scene) -> str | None   entrega un paquete a la estación de recogida y PARA.
                                   Devuelve su id, o None si no llegó nada.
    release(scene)                 vuelve a arrancar, cuando la mano ya no está encima.

── DE UNA FUENTE A TRES ─────────────────────────────────────────────────────────

La cabecera original de este fichero describía una sola cosa, la cinta. Aquí hay tres, y
son tres TAREAS distintas, no tres formas de hacer la misma:

    `table`     los bultos esperan colocados en la mesa. Es el caso base: la fuente no
                se mueve y el único problema es paletizar.
    `conveyor`  una banda entrega uno cada vez, singulado, siempre por el mismo sitio.
                Se para para que el robot coja. Puede atascarse.
    `truck`     el remolque entrega la carga ENTERA de golpe, apilada en columnas. El
                orden de extracción deja de venir dado: hay que deducirlo de lo que se
                ve. Ver `src/cell/truck.py`.

`present()`/`release()` sigue siendo el contrato; lo que cambia es que ahora tiene tres
implementaciones y un selector. El fichero conserva el nombre porque es el que el repo
designó para este contrato, y `table.py` y `truck.py` son hermanos suyos. **Esto es un
fichero donde la arquitectura hablaba de uno: está puesto en el PR para que se decida.**

── DOS AVISOS QUE NO SE VEN LEYENDO EL CÓDIGO ───────────────────────────────────

**Ninguna fuente tiene evento propio.** `events.kind` es un CHECK cerrado de seis
valores —`perceive | plan | pick | place | settle | fail`— y ninguno es suyo. Su estado
viaja en el `payload` del evento `perceive`, donde sobrar es inocuo, o no viaja.
Inventar un kind hace que la base rechace la fila con un 400 que el SDK se traga, y a
partir de ahí se apaga la subida del resto de la ejecución.

**Un atasco es `timeout`, no una causa nueva.** No hay valor en el vocabulario para "la
fuente no entregó". Si la estación se queda vacía y expira la espera, es `timeout`; si
entrega pero la percepción no ve nada, es `no_detection`. Son cosas distintas y el
gráfico de fallos las separa, así que no las mezcles. `exhausted` es la tercera y no es
un fallo: significa que no quedaban paquetes, que es como acaba un episodio bien.

**`release()` va DESPUÉS de que el brazo se haya retirado**, no antes: arrancar con la
mano dentro arrastra el siguiente paquete contra la herramienta.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# A qué altura sobre la banda se considera que un cartón va montado en ella. Por encima
# de esto ha volcado, se ha subido a otro o se ha caído: la banda deja de arrastrarlo y
# la espera acaba expirando, que es el camino físico a `timeout`.
BELT_BAND = 0.04


class Supply(Protocol):
    """De dónde salen los paquetes. Tres implementaciones, un contrato."""

    def stage(self, scene) -> None:
        """Pone la carga donde empieza y deja que se asiente. Una vez por episodio."""

    def present(self, scene) -> str | None:
        """Entrega el siguiente a la estación y para. `None` = no llegó nada."""

    def release(self, scene) -> None:
        """Vuelve a arrancar. Sólo con el brazo ya retirado."""

    def forecast(self, scene) -> list[str]:
        """Lo que viene detrás, si se puede saber."""


class _BaseSupply:
    """Lo que comparten las tres: la cola, y saber distinguir vacío de atascado."""

    def __init__(self, scene):
        self.pending: list[int] = [box.index for box in scene.boxes]
        self.current: int | None = None
        # Se acabó la carga. NO es un fallo: es como termina un episodio que salió bien.
        self.exhausted = False
        # La estación se quedó vacía y expiró la espera -> `timeout`.
        self.jammed = False

    def release(self, scene) -> None:
        """El brazo ya se ha llevado el de la estación."""
        if self.current is not None:
            self.pending = [i for i in self.pending if i != self.current]
            self.current = None

    def forecast(self, scene) -> list[str]:
        """Por defecto, nada: una fuente singulada sólo enseña el siguiente."""
        return []

    def _state(self, scene) -> dict:
        """Lo que la fuente quiere contar de sí misma, para el payload de `perceive`.

        Va ahí porque en el payload sobrar es inocuo y porque la fuente no tiene evento
        propio. Ver la cabecera.
        """
        return {
            "source": scene.level.source,
            "pending": len(self.pending),
            "exhausted": self.exhausted,
            "jammed": self.jammed,
        }


class Belt(_BaseSupply):
    """La cinta de verdad: una banda que arrastra el cartón y se para en la estación.

    MuJoCo no tiene primitiva de cinta. De las dos opciones que el repo daba por buenas
    —una banda con `slide` y actuador de velocidad, o aplicar velocidad a los cuerpos en
    contacto— se usa la segunda. La primera se queda sin recorrido a mitad de trayecto y
    habría que reponerla, y reponer la banda es teletransportar por la puerta de atrás.

    Se arrastra fijando la componente X de la velocidad de lo que va montado en la
    banda, y parar es poner la consigna a cero. El primer intento fue aplicar una fuerza
    de arrastre al cartón, que suena más físico y no lo es: la banda arrastra porque se
    mueve la SUPERFICIE, así que empujar el cartón contra una superficie quieta es
    pelearse con la fricción en vez de usarla. Medido: 8,4 N de empuje contra 37 N de
    fricción estática, el cartón no se movía y todos los episodios salían `timeout`.

    Lo demás —giro, asiento, contactos— lo sigue resolviendo la física. Y el atasco
    sigue siendo alcanzable de verdad: un cartón que vuelca o se sale de la banda deja
    de ir montado en ella, la banda deja de arrastrarlo, y la espera expira.

    El paquete se inyecta en la ENTRADA de la banda, no en la estación, y hace el
    trayecto entero: llega con la pose que traiga, torcida incluida. Inyectar en la
    entrada es modelar el alimentador de aguas arriba, que es lo que hay antes de la
    cinta en una planta de verdad.
    """

    def __init__(self, scene):
        super().__init__(scene)
        cfg = scene.cfg["conveyor"]
        self.speed = float(cfg["speed"])
        self.station_x, self.station_y = (float(v) for v in cfg["station"])
        self.tol = float(cfg["station_tol"])
        self.timeout_s = float(cfg["timeout_s"])
        self.surface_z = float(cfg["height"])
        length = float(cfg["dims"][0])
        self.entry_x = float(cfg["center"][0]) - length / 2 + 0.12
        self.running = False

    def stage(self, scene) -> None:
        """La banda empieza vacía: los paquetes están aguas arriba, fuera de escena."""
        for box in scene.boxes:
            scene.park_box(box.index)
        scene.settle(0.1)

    def _inject(self, scene, index: int) -> None:
        """Deja el siguiente cartón en la ENTRADA de la banda, con su desvío."""
        box = scene.boxes[index]
        jitter = scene.level.pos_jitter_m
        yaw = np.radians(scene.level.yaw_jitter_deg) * scene.rng.uniform(-1.0, 1.0)
        offset = scene.rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        scene.place_box(
            index,
            (self.entry_x, self.station_y + offset, self.surface_z + box.dims_m[2] / 2 + 0.002),
            yaw,
        )
        scene.settle(0.15)

    def _riding(self, scene, index: int) -> bool:
        """Si ese cartón va montado en la banda ahora mismo."""
        box = scene.boxes[index]
        z = float(scene.data.xpos[scene.body_id(index)][2])
        return abs(z - (self.surface_z + box.dims_m[2] / 2)) < BELT_BAND

    def _drive(self, scene) -> None:
        """Un tick de banda: arrastra hacia la estación lo que vaya montado en ella."""
        for index in self.pending:
            if not self._riding(scene, index):
                continue
            if float(scene.data.xpos[scene.body_id(index)][0]) > self.station_x:
                continue
            joint = scene.mujoco.mj_name2id(
                scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, scene.boxes[index].joint
            )
            if joint < 0:
                continue
            dof = scene.model.jnt_dofadr[joint]
            scene.data.qvel[dof] = self.speed

    def _stop(self, scene) -> None:
        """Consigna a cero. Lo que va montado se para con la banda."""
        for index in self.pending:
            joint = scene.mujoco.mj_name2id(
                scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, scene.boxes[index].joint
            )
            if joint >= 0:
                dof = scene.model.jnt_dofadr[joint]
                scene.data.qvel[dof : dof + 3] = 0.0
        self.running = False

    def present(self, scene) -> str | None:
        if not self.pending:
            self.exhausted = True
            return None
        index = self.pending[0]
        self.current = index
        self._inject(scene, index)

        self.running = True
        deadline = scene.clock + self.timeout_s
        body = scene.body_id(index)
        while scene.clock < deadline:
            self._drive(scene)
            scene.step(1.0 / scene.cfg["episode"]["control_hz"])
            if abs(float(scene.data.xpos[body][0]) - self.station_x) < self.tol:
                self._stop(scene)
                scene.settle(0.25)            # que deje de rodar antes de mirarlo
                return scene.boxes[index].package_id

        # La banda empujó lo que pudo y el cartón no llegó: atasco.
        self._stop(scene)
        self.jammed = True
        self.current = None
        return None


class TableSupply(_BaseSupply):
    """Los bultos esperan colocados en la mesa. La fuente no se mueve.

    Es el caso base y el que hacía el repo guionizado. Sigue sin ser trivial: el nivel
    puede dejarlos girados y descentrados, así que la herramienta tiene que alinearse
    con el cartón que va a sellar y no con la mesa.
    """

    def stage(self, scene) -> None:
        cfg = scene.cfg["table"]
        center_x, center_y = (float(v) for v in cfg["center"])
        half_x, half_y, _ = (float(v) for v in cfg["size"])
        top = float(cfg["height"])
        jitter = scene.level.pos_jitter_m
        spread = np.radians(scene.level.yaw_jitter_deg)

        # Rejilla suelta sobre la mesa: caben en dos filas sin tocarse, y el nivel les
        # añade su desvío encima.
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
        """El siguiente de la mesa. No hay nada que mover, así que no puede atascarse."""
        if not self.pending:
            self.exhausted = True
            return None
        self.current = self.pending[0]
        return scene.boxes[self.current].package_id


def make_supply(scene) -> Supply:
    """La fuente que pide el nivel. Es la única elección que hace la tarea."""
    from src.cell.truck import TruckSupply

    return {
        "table": TableSupply,
        "conveyor": Belt,
        "truck": TruckSupply,
    }[scene.level.source](scene)


# El contrato, también como funciones libres: es como lo declaraba la arquitectura y hay
# código escrito contra esa forma.
def present(scene) -> str | None:
    """Entrega un paquete a la estación y para. `None` = no llegó nada."""
    if getattr(scene, "supply", None) is None:
        scene.supply = make_supply(scene)
        scene.supply.stage(scene)
    return scene.supply.present(scene)


def release(scene) -> None:
    """Vuelve a arrancar la fuente. Sólo con el brazo ya retirado."""
    if getattr(scene, "supply", None) is not None:
        scene.supply.release(scene)
