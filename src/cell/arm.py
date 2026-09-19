"""
El control del brazo: IK cartesiana, ventosa y bucle de física.

Portado desde `tools/stable_pallet/simulator.py`. `ArmController` expone:

    tcp_pose()                 pose actual del TCP
    move_to(pose)              waypoint cartesiano; False si no converge -> ik_unreachable
    move_joints(qpos, dur)     espacio de juntas; SOLO para la foto final
    prepare_grasp(grasp)       fija el bloque de ventosas ANTES de bajar a sellar
    seal(index)                sella las ventosas sobre una caja; False -> grasp_slip
    release()                  corta el vacío
    is_holding()               False si la caja se escurrió -> grasp_slip
    step_physics()             un tick de control; es lo que refresca el visor
    steps_per_tick

── UN CAMBIO RESPECTO A LA CABECERA ORIGINAL ────────────────────────────────────

Era `set_gripper(value, hold_s)` con un recorrido 0-255 y `is_holding(width)` con la
apertura esperada. Eso es una pinza paralela. Aquí la herramienta es un OnRobot VGP20 de
16 ventosas: no hay recorrido que mandar ni apertura que comprobar, hay un bloque de
ventosas que sellar y un vacío que cortar. `is_holding()` pierde el argumento porque ya
no hay una anchura de referencia: o la caja sigue pegada o no.

── TRES COSAS QUE CUESTA CARO VOLVER A DESCUBRIR ────────────────────────────────

**La tolerancia de convergencia va POR ENCIMA del suelo medido del servo.** Un servo de
posición aguanta contra la gravedad con error permanente de varios miliradianes, y un
cartón sobre las ventosas lo empuja más allá de quince. Esperar a que las juntas
alcancen la consigna es esperar para siempre. `move_to` mide y corrige el error EN
CARTESIANO, y el bucle de reposo sólo contesta a la otra pregunta: ¿ha dejado de
moverse el brazo? Los números y su porqué, en `configs/pallet.yaml`.

**Todo el trayecto va en cartesiano a una cota de tránsito fija**: subir recto, cruzar,
bajar recto. El rodeo por `home` en espacio de juntas tira la caja al ir y barre el
montón al volver. `move_joints` existe sólo para la foto final, cuando ya no queda nada
que colocar.

**El bloque de ventosas que agarra una caja estrecha está DESCENTRADO respecto al
cuerpo**, y el brazo tiene que cancelar ese desplazamiento en cada movimiento desde el
acercamiento de agarre, no solo con el cartón ya en la mano. Si se centra el TCP, se
sella y *después* se resta el offset, el cartón viaja el desplazamiento entero al
primer waypoint: un `book_s` sale ~45 mm corrido. La rejilla vive en el frame de la
herramienta, que apunta hacia abajo, así que cancelarlo directamente en coordenadas de
mundo lo DUPLICA en un eje en vez de quitarlo y la fila exterior de ventosas queda
colgando por fuera del cartón. Ver `TOOL_DOWN`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.cell import Pose, grasp_offset_world, tcp_frame
from src.cell.scene import HeldPackage


@dataclass(frozen=True)
class Grasp:
    """Con qué ventosas se coge una caja, y si eso llega para levantarla."""

    cups: tuple[tuple[float, float], ...]
    capacity_newtons: float
    required_newtons: float
    tool_offset: tuple[float, float]

    @property
    def feasible(self) -> bool:
        return bool(self.cups) and self.capacity_newtons >= self.required_newtons

    @property
    def capacity_ratio(self) -> float:
        return self.capacity_newtons / self.required_newtons


class VacuumArray:
    """La rejilla de ventosas del OnRobot VGP20.

    Abarca los 264 x 184 mm del cuerpo entero, así que un cartón más corto que 264 mm no
    llega a las columnas exteriores y uno menos hondo que 184 mm no llega a las filas de
    fuera. Centrar el CUERPO sobre un cartón así desperdicia las ventosas que caen por
    fuera de su borde; `plan` escoge en cambio el mayor bloque contiguo que cabe y dice
    qué desplazamiento pone ESE bloque —no el cuerpo— sobre el centro del cartón.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["vacuum"]

    def _axis_block(self, count: int, pitch: float, span: float):
        """El tramo más largo de ventosas que cabe en `span`, y dónde queda su centro.

        Los empates van al tramo más cercano al eje de la herramienta, para que la carga
        se quede junto a la brida en vez de colgar de un lado.
        """
        usable = span / 2 - self.cfg["cup_radius"]
        coords = [(i - (count - 1) / 2) * pitch for i in range(count)]
        best: tuple[float, ...] = ()
        best_centre = 0.0
        for start in range(count):
            for end in range(start, count):
                if (coords[end] - coords[start]) / 2 > usable:
                    continue
                run = tuple(coords[start : end + 1])
                centre = (coords[start] + coords[end]) / 2
                if len(run) > len(best) or (len(run) == len(best) and abs(centre) < abs(best_centre)):
                    best, best_centre = run, centre
        return best, best_centre

    def plan(self, dims_m, mass_kg: float) -> Grasp:
        width, depth, _ = dims_m
        xs, offset_x = self._axis_block(self.cfg["columns"], self.cfg["pitch_x"], width)
        ys, offset_y = self._axis_block(self.cfg["rows"], self.cfg["pitch_y"], depth)
        cups = tuple((x, y) for x in xs for y in ys)
        capacity = len(cups) * self.cfg["cup_force_newtons"] * self.cfg["derating"]
        required = mass_kg * 9.81 * self.cfg["safety_factor"]
        return Grasp(cups, capacity, required, (offset_x, offset_y))


class UnreachablePose(RuntimeError):
    """El brazo no puede llegar, o no puede sostenerse, donde se le ha pedido."""


class ArmController:
    """El UR10e con su ventosa, sobre una `PalletScene` ya montada."""

    def __init__(self, scene):
        self.scene = scene
        self.mujoco = scene.mujoco
        motion = scene.cfg["motion"]
        self.motion = motion
        self.vacuum = VacuumArray(scene.cfg)
        self.cup_gap = float(scene.cfg["vacuum"]["cup_gap"])
        self.reach_tolerance = float(motion["reach_tolerance"])
        self.fast_forward = False
        self.held: int | None = None
        self.grasp: Grasp | None = None
        # Lo que quedó sin corregir en el último `move_to`, en metros. Un acierto limpio
        # y un roce justo por debajo de `reach_tolerance` acaban los dos en `True`, así
        # que sin esto no hay forma de distinguirlos desde fuera.
        self.last_residual = 0.0
        # La orientación de trabajo tal y como el modelo la ve. Se lee una vez, con el
        # brazo en la pose de reposo, y es la referencia de todas las poses del episodio.
        self.reference = scene.data.site_xmat[scene.tcp_site].reshape(3, 3).copy()

    # ── estado ───────────────────────────────────────────────────────────────

    @property
    def steps_per_tick(self) -> int:
        return self.scene.steps_per_tick

    def tcp_pose(self) -> Pose:
        scene = self.scene
        return Pose(
            scene.data.site_xpos[scene.tcp_site].copy(),
            scene.data.site_xmat[scene.tcp_site].reshape(3, 3).copy(),
        )

    def step_physics(self) -> None:
        """Un tick de control. Es lo que refresca el visor."""
        for _ in range(self.steps_per_tick):
            self.mujoco.mj_step(self.scene.model, self.scene.data)
            self.scene.clock += self.scene.model.opt.timestep
        self.scene.sync_viewer()

    # ── cinemática ───────────────────────────────────────────────────────────

    @staticmethod
    def _rotation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Vector de error de orientación, en el frame del mundo."""
        error = desired @ current.T
        return 0.5 * np.array([
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ])

    def solve_ik(self, position, rotation) -> np.ndarray | None:
        """Mínimos cuadrados amortiguados sobre `mj_jacSite`. `None` = no converge.

        Se prueba desde la pose actual y desde la de reposo: la primera da trayectos
        cortos y continuos, la segunda rescata los casos en que la actual está en una
        rama de la que no se sale.
        """
        scene, cfg = self.scene, self.motion
        target = np.asarray(position, dtype=float)
        jac_pos = np.zeros((3, scene.model.nv))
        jac_rot = np.zeros((3, scene.model.nv))
        current = np.asarray(scene.data.qpos[scene.arm_qpos]).copy()

        for seed in (current, scene.home.copy()):
            q = seed.copy()
            for _ in range(int(cfg["ik_iterations"])):
                scene.ik_data.qpos[scene.arm_qpos] = q
                self.mujoco.mj_forward(scene.model, scene.ik_data)
                position_error = target - scene.ik_data.site_xpos[scene.tcp_site]
                rotation_error = self._rotation_error(
                    rotation, scene.ik_data.site_xmat[scene.tcp_site].reshape(3, 3)
                )
                if (np.linalg.norm(position_error) < cfg["ik_tol_pos"]
                        and np.linalg.norm(rotation_error) < cfg["ik_tol_rot"]):
                    return q
                self.mujoco.mj_jacSite(scene.model, scene.ik_data, jac_pos, jac_rot, scene.tcp_site)
                jacobian = np.vstack((jac_pos[:, scene.arm_dofs], jac_rot[:, scene.arm_dofs]))
                error = np.concatenate((position_error, rotation_error))
                damping = float(cfg["ik_damping"])
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(6), error
                )
                q += np.clip(delta, -cfg["ik_step_limit"], cfg["ik_step_limit"])
                q[2] = np.clip(q[2], -3.13, 3.13)          # el codo tiene rango limitado
                q = (q + math.pi) % (2 * math.pi) - math.pi
        return None

    def _duration(self, distance: float, approach: bool) -> float:
        """Cuánto debe durar un waypoint. Tránsito y descenso NO van a la misma velocidad."""
        cfg = self.motion
        speed = cfg["approach_speed"] if approach else cfg["cartesian_speed"]
        return float(np.clip(distance / speed, cfg["min_move_s"], cfg["max_move_s"]))

    def move_joints(self, target_q, seconds: float = 0.85) -> None:
        """Interpola en espacio de juntas y espera a que el brazo se quede quieto.

        SOLO para la foto final. En espacio de juntas el recorrido no está controlado, y
        el brazo acaba justo encima del palé: medido, meter la foto por capa con este
        movimiento bajaba el episodio de 10/10 a 4/10 con `overhang_violation`.
        """
        scene, cfg = self.scene, self.motion
        start_q = np.asarray(scene.data.qpos[scene.arm_qpos], dtype=float).copy()
        target_q = np.asarray(target_q, dtype=float)
        steps = max(1, round(seconds / scene.model.opt.timestep))
        for step in range(steps):
            phase = (step + 1) / steps
            scene.data.ctrl[:] = start_q + (target_q - start_q) * phase * phase * (3 - 2 * phase)
            self.mujoco.mj_step(scene.model, scene.data)
            scene.clock += scene.model.opt.timestep
            scene.sync_viewer()
        scene.data.ctrl[:] = target_q

        # Esperar a que las juntas ALCANCEN la consigna es esperar para siempre: el servo
        # aguanta contra la gravedad con error permanente, y un cartón en las ventosas lo
        # empuja más. Eso es justo lo que `move_to` mide y corrige en cartesiano, así que
        # este bucle sólo contesta a la otra pregunta: ¿ha dejado de moverse?
        still = 0
        for _ in range(int(cfg["rest_timeout"])):
            if np.max(np.abs(scene.data.qvel[scene.arm_dofs])) < cfg["rest_speed"]:
                still += 1
                if still >= cfg["rest_steps"]:
                    break
            else:
                still = 0
            self.mujoco.mj_step(scene.model, scene.data)
            scene.clock += scene.model.opt.timestep
            scene.sync_viewer()

    def _snap_to(self, target_q) -> None:
        """Pone el brazo en una pose resuelta sin recorrer el camino.

        El fast-forward se salta el tránsito, NO la física que decide el resultado: la
        celda sigue asentándose al llegar, así que la caída del servo, los contactos y la
        suelta ocurren igual que tras un movimiento de verdad.
        """
        scene = self.scene
        scene.data.qpos[scene.arm_qpos] = target_q
        scene.data.qvel[scene.arm_dofs] = 0.0
        scene.data.ctrl[:] = target_q
        self.mujoco.mj_forward(scene.model, scene.data)
        self._carry_held()
        scene.step(0.06)

    def move_to(self, pose: Pose, *, approach: bool = False) -> bool:
        """Lleva el TCP a `pose`. `False` = no llega, y eso es `ik_unreachable`.

        Tres pasadas de corrección: se resuelve, se va, se mide el error que queda en
        cartesiano y se vuelve a pedir compensándolo. Es lo que quita la caída
        estacionaria del servo bajo carga sin tener que modelarla.
        """
        scene = self.scene
        desired = np.asarray(pose.position, dtype=float)
        commanded = desired.copy()
        offset = self._held_offset_world(pose)
        desired_tool = desired - offset
        commanded = desired_tool.copy()

        for attempt in range(3):
            target_q = self.solve_ik(commanded, pose.rotation)
            if target_q is None:
                return False
            if self.fast_forward:
                self._snap_to(target_q)
            else:
                distance = float(np.linalg.norm(
                    scene.data.site_xpos[scene.tcp_site] - commanded))
                seconds = self._duration(distance, approach) if attempt == 0 else 0.28
                self.move_joints(target_q, seconds)
            error = desired_tool - scene.data.site_xpos[scene.tcp_site]
            if np.linalg.norm(error) < 0.0025:
                self.last_residual = float(np.linalg.norm(error))
                return True
            commanded = commanded + error

        # Una pose que el solver acepta no es una pose que el brazo pueda SOSTENER:
        # plegado en corto, los eslabones se apoyan en el propio pedestal y los servos se
        # quedan cortos. Sellar sobre un cartón al que la herramienta nunca llegó es como
        # una celda lo coge por una esquina y lo deja en otro sitio.
        residual = float(np.linalg.norm(desired_tool - scene.data.site_xpos[scene.tcp_site]))
        self.last_residual = residual
        return residual <= self.reach_tolerance

    # ── la ventosa ───────────────────────────────────────────────────────────

    def _held_offset_world(self, pose: Pose) -> np.ndarray:
        """Dónde queda el bloque de ventosas activo respecto al origen de la herramienta.

        Cero si aún no hay un `grasp` elegido. Con uno fijado —antes de sellar, con el
        cartón en la mano, da igual— es lo que hay que restar al destino para que el
        CARTÓN acabe donde se pidió, y no la brida. Se elige en `_pick` antes de bajar:
        aplicarlo solo después de `seal` desplaza el paquete el offset entero.
        """
        if self.grasp is None:
            return np.zeros(3)
        return grasp_offset_world(self.grasp.tool_offset, pose.yaw, self.reference)

    def carton_xy(self) -> tuple[float, float]:
        """XY del cartón que `move_to` está sirviendo, no de la brida.

        Con un paquete agarrado es la pose real del cuerpo. Sin él, TCP más el offset
        del bloque, que es lo que hay que usar para retirar en vertical tras soltar.
        """
        if self.held is not None:
            position, _ = self.scene.box_pose(self.held)
            return float(position[0]), float(position[1])
        pose = self.tcp_pose()
        world = pose.position + self._held_offset_world(pose)
        return float(world[0]), float(world[1])

    def plan_grasp(self, box) -> Grasp:
        """Qué ventosas cubren esta caja y si el vacío llega para levantarla."""
        return self.vacuum.plan(box.dims_m, box.mass_kg)

    def prepare_grasp(self, grasp: Grasp) -> None:
        """Fija el bloque de ventosas que `move_to` pone sobre el centro del cartón.

        Tiene que ocurrir ANTES del descenso de agarre. Si se sella con el cuerpo
        centrado y el offset se aplica después, el cartón viaja el desplazamiento
        entero al primer waypoint y un `book_s` sale ~45 mm corrido.
        """
        self.grasp = grasp
        self.show_cups(grasp)

    def show_cups(self, grasp: Grasp) -> None:
        """Enciende en el visor las ventosas que están sellando. Sólo es color."""
        scene = self.scene
        active = {(round(x, 4), round(y, 4)) for x, y in grasp.cups}
        cfg = scene.cfg["vacuum"]
        for column in range(cfg["columns"]):
            x = (column - (cfg["columns"] - 1) / 2) * cfg["pitch_x"]
            for row in range(cfg["rows"]):
                y = (row - (cfg["rows"] - 1) / 2) * cfg["pitch_y"]
                geom = self.mujoco.mj_name2id(
                    scene.model, scene.mujoco.mjtObj.mjOBJ_GEOM, f"cup_{column}_{row}"
                )
                if geom < 0:
                    continue
                scene.model.geom_rgba[geom] = (
                    [0.10, 0.85, 0.90, 1.0] if (round(x, 4), round(y, 4)) in active
                    else [0.18, 0.22, 0.24, 0.22]
                )

    def _equality(self, index: int) -> int:
        return self.mujoco.mj_name2id(
            self.scene.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{index}"
        )

    def seal(self, index: int, grasp: Grasp | None = None) -> bool:
        """Sella las ventosas sobre la caja `index`. `False` = no agarra -> `grasp_slip`.

        La unión se fija con la transformada RELATIVA medida en este instante, no con una
        nominal: la caja está donde está, torcida incluida, y lo que se agarra es eso.
        """
        scene = self.scene
        box = scene.boxes[index]
        grasp = self.plan_grasp(box) if grasp is None else grasp
        if not grasp.feasible:
            return False

        equality = self._equality(index)
        if equality < 0:
            return False
        body = scene.body_id(index)
        tool_rotation = scene.data.xmat[scene.tool_body].reshape(3, 3)
        relative = tool_rotation.T @ (scene.data.xpos[body] - scene.data.xpos[scene.tool_body])
        inverse_tool = np.empty(4)
        relative_quat = np.empty(4)
        self.mujoco.mju_negQuat(inverse_tool, scene.data.xquat[scene.tool_body])
        self.mujoco.mju_mulQuat(relative_quat, inverse_tool, scene.data.xquat[body])
        scene.model.eq_data[equality, 3:6] = relative
        scene.model.eq_data[equality, 6:10] = relative_quat
        scene.data.eq_active[equality] = 1
        self.mujoco.mj_forward(scene.model, scene.data)

        self.held = index
        self.grasp = grasp
        self.show_cups(grasp)
        scene.step(0.15)
        return self.is_holding()

    def make_rigid(self) -> bool:
        """Reparenta el paquete sellado para que el sensor de muñeca vea un agarre rígido.

        Una `weld` sesga el par que informa la brida —de 50 a 750 mm de error de CoM a
        estas masas, y esperar no lo quita. Un sello de ventosa es una unión rígida: el
        cartón cuelga de `tool` y la restricción desaparece.
        """
        if self.held is None:
            return False
        scene = self.scene
        if scene.held is not None and scene.held.index == self.held:
            return self.is_holding()
        index = self.held
        body = scene.body_id(index)
        tool_rotation = scene.data.xmat[scene.tool_body].reshape(3, 3)
        position = tool_rotation.T @ (scene.data.xpos[body] - scene.data.xpos[scene.tool_body])
        inverse_tool = np.empty(4)
        relative = np.empty(4)
        self.mujoco.mju_negQuat(inverse_tool, scene.data.xquat[scene.tool_body])
        self.mujoco.mju_mulQuat(relative, inverse_tool, scene.data.xquat[body])
        scene.rebuild(HeldPackage(
            index,
            tuple(float(value) for value in position),
            tuple(float(value) for value in relative),
        ))
        if self.grasp is not None:
            self.show_cups(self.grasp)
        return self.is_holding()

    def release(self) -> None:
        """Corta el vacío. Con ventosa no hay que abrir nada: se suelta y ya."""
        if self.held is None:
            return
        if self.scene.held is not None:
            self.scene.rebuild(None)
        else:
            equality = self._equality(self.held)
            if equality >= 0:
                self.scene.data.eq_active[equality] = 0
                self.mujoco.mj_forward(self.scene.model, self.scene.data)
        self.held = None
        self.grasp = None
        self.scene.step(float(self.scene.cfg["motion"].get("release_settle_s", 0.15)))

    def is_holding(self) -> bool:
        """`False` si la caja se escurrió de las ventosas -> `grasp_slip`.

        No basta con que la restricción siga activa: se comprueba que la caja siga donde
        la herramienta cree que está. Una unión activa sobre un cartón que se ha ido es
        justo el fallo que hay que poder contar.
        """
        if self.held is None:
            return False
        scene = self.scene
        box = scene.boxes[self.held]
        cup_face = scene.data.site_xpos[scene.tcp_site]
        top = scene.box_top_center(self.held)
        return float(np.linalg.norm(cup_face[:2] - top[:2])) < max(box.dims_m[0], box.dims_m[1])

    def _carry_held(self) -> None:
        """Arrastra la caja sellada al saltar de pose, para que no se quede atrás."""
        if self.held is None:
            return
        scene = self.scene
        equality = self._equality(self.held)
        if equality < 0 or not scene.data.eq_active[equality]:
            return
        tool_position = scene.data.xpos[scene.tool_body]
        tool_rotation = scene.data.xmat[scene.tool_body].reshape(3, 3)
        relative = scene.model.eq_data[equality, 3:6]
        relative_quat = scene.model.eq_data[equality, 6:10]
        world_quat = np.empty(4)
        self.mujoco.mju_mulQuat(world_quat, scene.data.xquat[scene.tool_body], relative_quat)
        scene._write_free_pose(self.held, tool_position + tool_rotation @ relative, world_quat)
        self.mujoco.mj_forward(scene.model, scene.data)

    # ── maniobras ────────────────────────────────────────────────────────────

    def go_to(self, x: float, y: float, z: float, yaw: float, *, approach: bool = False) -> bool:
        """Azúcar: un waypoint cartesiano desde coordenadas sueltas."""
        return self.move_to(tcp_frame((x, y, z), yaw), approach=approach)

    def park(self) -> bool:
        """Aparta el brazo del encuadre, EN CARTESIANO, para poder sacar la foto.

        Acaba justo encima del palé, que es donde estaba soltando, así que sin esto la
        cenital sale del dorso de la mano. Y tiene que ser cartesiano: en espacio de
        juntas el recorrido no está controlado y barre el montón recién colocado.
        """
        motion = self.scene.cfg["motion"]
        x, y = motion["park_xy"]
        return self.go_to(float(x), float(y), float(motion["park_z"]), 0.0)

    def go_home(self) -> None:
        """A la pose de reposo, en espacio de juntas. Sólo al final del episodio."""
        self.move_joints(self.scene.observe_qpos, 1.2)
