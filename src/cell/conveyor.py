"""
La fuente de suministro: de dónde sale el paquete que el brazo tiene que coger.

    present(scene) -> str | None   entrega un paquete a la estación de recogida y PARA.
                                   Devuelve su id, o None si no llegó nada.
    release(scene)                 vuelve a arrancar, cuando la mano ya no está encima.

── DE UNA FUENTE A TRES ─────────────────────────────────────────────────────────

La cabecera original de este fichero describía una sola cosa, la cinta. Aquí hay tres, y
son tres TAREAS distintas, no tres formas de hacer la misma:

    `table`     un ascensor y una banda entregan los bultos en el centro de la mesa.
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
el cartón llega y no se asienta, también. Si entrega pero la percepción no ve nada, es
`no_detection`. Son cosas distintas y el gráfico de fallos las separa, así que no las
mezcles. `exhausted` es la tercera y no es un fallo: significa que no quedaban paquetes,
que es como acaba un episodio bien.

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

# Reposo medido en la estación, sobre el centro de la tapa: es la pose en la que
# `_pick` sella, no el `qvel` crudo. Un settle fijo de 0.25 s devolvía `present()`
# con el cartón aún botando: nivel 22, semilla 1, `std_m-00` salía con vx≈-0.019 m/s
# y wy≈-0.17 rad/s; el centro de la tapa seguía a ~0.30 m/s (p95) y en el segundo
# siguiente se movía 47 mm en X y 36 mm en Z. Esperar 1 s más dejaba esa pose quieta
# y bajaba el error XY de 28 mm a 2.4 mm.
#
# El `qvel` no sirve como umbral único: un `book_s` asentado en la banda mantiene
# |ω| de contacto de hasta 0.12 rad/s con 0.1 mm de deriva neta en 2 s, y expirar
# entonces es un `timeout` falso. Estos umbrales están entre ese chatter (p95 del
# centro de la tapa ≈ 9 mm/s) y el bote del bug (p95 ≈ 300 mm/s). No se toca
# `episode.tolerance_xy` para enmascarar el movimiento.
REST_LIN_VEL = 0.015              # m/s del centro de la tapa
REST_HOLD_S = 0.10                # 5 ticks a 50 Hz; un cruce por cero no es quietud


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

    Lo demás —giro, asiento, contactos— lo sigue resolviendo la física. Llegar a la
    estación no basta: `present()` no devuelve hasta que el centro de la tapa está
    quieto, o hasta que expira `timeout_s`. Y el atasco sigue siendo alcanzable de
    verdad: un cartón que vuelca, se sale de la banda o no se asienta deja de contar
    como entregado, y la espera expira.

    El paquete se carga abajo en el ascensor. Su plataforma mocap sube por una rampa
    suave y el contacto eleva el cartón; sólo después arranca la banda. La carga inicial
    es la única escritura de pose: desde ahí llega por física, torcida incluida.
    """

    def __init__(self, scene):
        super().__init__(scene)
        # El carril lo decide la FUENTE del nivel, no esta clase: la cinta entrega bajo
        # el brazo y la mesa entrega a la mesa, y por lo demás son la misma banda. Ver
        # `scene.lane_for` y `src/cell/table.py`.
        from src.cell.scene import lane_for

        cfg = lane_for(scene.cfg, scene.level.source) or scene.cfg["conveyor"]
        self.speed = float(cfg["speed"])
        self.station_x, self.station_y = (float(v) for v in cfg["station"])
        self.tol = float(cfg["station_tol"])
        self.timeout_s = float(cfg["timeout_s"])
        self.surface_z = float(cfg["height"])
        length = float(cfg["dims"][0])
        self.entry_x = float(cfg["center"][0]) - length / 2 - float(scene.cfg["elevator"]["depth"]) / 2
        self.lower_height = float(scene.cfg["elevator"]["lower_height"])
        self.lift_speed = float(scene.cfg["elevator"]["speed"])
        self.running = False

    def stage(self, scene) -> None:
        """La línea empieza vacía: los paquetes esperan dentro de la caja negra."""
        for box in scene.boxes:
            scene.park_box(box.index)
        scene.settle(0.1)

    def _move_elevator(self, scene, top: float, deadline: float) -> bool:
        """Mueve el soporte; la caja sube por contacto, sin escribir su pose ni velocidad."""
        mocap = int(scene.model.body_mocapid[scene.model.body("elevator").id])
        half_height = float(scene.model.geom("elevator_platform").size[2])
        start = float(scene.data.mocap_pos[mocap, 2])
        distance = top - half_height - start
        step = float(scene.model.opt.timestep)
        # Curva suave: velocidad cero en los extremos y pico igual a la consigna.
        ticks = max(1, int(np.ceil(1.5 * abs(distance) / self.lift_speed / step)))
        for tick in range(1, ticks + 1):
            if scene.clock >= deadline:
                return False
            fraction = tick / ticks
            scene.data.mocap_pos[mocap, 2] = start + distance * fraction**2 * (3 - 2 * fraction)
            scene.step(step)
        return True

    def _inject(self, scene, index: int, deadline: float) -> bool:
        """Carga abajo, eleva por contacto y entrega a la banda a la misma cota."""
        if not self._move_elevator(scene, self.lower_height, deadline):
            return False
        box = scene.boxes[index]
        jitter = scene.level.pos_jitter_m
        yaw = np.radians(scene.level.yaw_jitter_deg) * scene.rng.uniform(-1.0, 1.0)
        offset = scene.rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        scene.place_box(
            index,
            (self.entry_x, self.station_y + offset, self.lower_height + box.dims_m[2] / 2 + 0.002),
            yaw,
        )
        if scene.clock + 0.15 > deadline:
            return False
        scene.settle(0.15)
        if not self._move_elevator(scene, self.surface_z, deadline):
            return False
        if scene.clock + 0.15 > deadline:
            return False
        scene.settle(0.15)
        return scene.clock < deadline and self._riding(scene, index)

    def _riding(self, scene, index: int) -> bool:
        """Si ese cartón va montado en la banda ahora mismo."""
        box = scene.boxes[index]
        z = float(scene.data.xpos[scene.body_id(index)][2])
        return abs(z - (self.surface_z + box.dims_m[2] / 2)) < BELT_BAND

    def _drive(self, scene) -> None:
        """Un PASO de banda: arrastra hacia la estación lo que vaya montado en ella.

        Va por paso de física, no por tick de control, y eso no es afinado: la superficie
        de la banda es estática y lo único que mueve el cartón es que se le reponga la
        velocidad. Entre reposición y reposición la fricción se la come, así que
        reponiéndola cada 20 ms —un tick de control, diez pasos— la consigna de 0.25 m/s
        se quedaba en 0.10 m/s reales: 11.9 s para los 1.23 m del trayecto, con el brazo
        parado mirando todo ese rato. Reponiéndola cada 2 ms salen 0.231 m/s y 5.3 s.
        Medido en el nivel 21.
        """
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

    def _resting(self, scene, index: int, previous_top: np.ndarray, dt: float) -> tuple[bool, np.ndarray]:
        """Si el centro de la tapa ya no se mueve. Devuelve también la pose nueva."""
        top = scene.box_top_center(index).copy()
        speed = float(np.linalg.norm(top - previous_top) / dt) if dt > 0 else 0.0
        return speed < REST_LIN_VEL, top

    def _wait_until_still(self, scene, index: int, deadline: float) -> bool:
        """Espera a reposo sostenido del centro de la tapa. `False` → `timeout`."""
        tick = 1.0 / scene.cfg["episode"]["control_hz"]
        still = 0.0
        previous = scene.box_top_center(index).copy()
        while scene.clock < deadline:
            scene.step(tick)
            resting, previous = self._resting(scene, index, previous, tick)
            if resting:
                still += tick
                if still >= REST_HOLD_S:
                    return True
            else:
                still = 0.0
        return False

    def _fail_delivery(self, scene) -> None:
        """No hubo entrega: atasco, y el director lo traduce a `timeout`."""
        self._stop(scene)
        self.jammed = True
        self.current = None

    def present(self, scene) -> str | None:
        if not self.pending:
            self.exhausted = True
            return None
        index = self.pending[0]
        self.current = index
        deadline = scene.clock + self.timeout_s
        if not self._inject(scene, index, deadline):
            self._fail_delivery(scene)
            return None

        self.running = True
        body = scene.body_id(index)
        step = scene.model.opt.timestep
        while scene.clock < deadline:
            self._drive(scene)
            scene.step(step)
            if abs(float(scene.data.xpos[body][0]) - self.station_x) < self.tol:
                self._stop(scene)
                if not self._wait_until_still(scene, index, deadline):
                    self._fail_delivery(scene)
                    return None
                return scene.boxes[index].package_id

        # La banda empujó lo que pudo y el cartón no llegó: atasco.
        self._fail_delivery(scene)
        return None


def make_supply(scene) -> Supply:
    """La fuente que pide el nivel. Es la única elección que hace la tarea."""
    from src.cell.table import TableSupply
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
