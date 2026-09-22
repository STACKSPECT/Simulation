"""Experimento separado: orientar cada cubo con el CoM hacia su apoyo.

No es otro nivel del paletizado ni escribe telemetría. Reutiliza la mesa de fuente, la
mesa auxiliar, el UR10e y el ``WristGauge``, pero tiene su propio director porque la
maniobra incluye una segunda recogida:

    mesa de fuente -> medir en la mano -> tumbar en la auxiliar
    -> volver a coger por arriba -> colocar en el palé

El gauge a plomo mide las dos componentes horizontales del CoM y no la vertical. Por
eso este experimento compara las cuatro caras laterales, no finge conocer cuál de las
caras superior e inferior está más cerca. Los cuatro cubos configurados tienen un sesgo
lateral inequívoco.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.cell import Pose
from src.cell.arm import ArmController
from src.cell.scene import Box, Level, PalletScene, load_configs
from src.cell.table import TableSupply
from src.contracts import Detector, Gauge, Observation, PackageSpec


class _StationSupply(TableSupply):
    """La mesa de fuente, pero dejando cada cubo YA en la estación, sin banda.

    La banda arrastra imponiendo velocidad al bulto contra una superficie de fricción
    1.1, y un cubo de 300 mm vuelca con eso: su canto delantero aguanta mientras la
    fricción no pase de semilado / altura del CoM, que es 1.0 con el CoM centrado y
    0.78 con el `com_x_pos`, que lo lleva 45 mm hacia delante. Las cajas del catálogo
    son bajas y largas —una `std_m` aguanta hasta 2.3— y por eso los niveles no lo ven.
    Medido al fusionar esta rama sobre la banda con ascensor: el primer cubo subía a
    z = 0.775 inclinado unos 24°, dejaba de ir montado y expiraba el plazo.

    Bajar la fricción del cubo no sirve —MuJoCo combina con el máximo— y tocar la de la
    banda cambiaría los niveles medidos. El experimento no ensaya la entrega: se escribió
    y se midió contra una mesa que ponía el bulto en su centro, que es la misma estación
    de hoy. Eso es lo que se hace aquí; el aparcamiento, la cola y `release` siguen
    siendo los de la mesa.
    """

    def present(self, scene) -> str | None:
        if not self.pending:
            self.exhausted = True
            return None
        index = self.pending[0]
        self.current = index
        height = scene.boxes[index].dims_m[2]
        scene.place_box(index, (self.station_x, self.station_y,
                                self.surface_z + height / 2 + 0.002))
        if not self._wait_until_still(scene, index, scene.clock + self.timeout_s):
            self._fail_delivery(scene)
            return None
        return scene.boxes[index].package_id


@dataclass(frozen=True)
class _OnPallet:
    """Un cubo ya en el palé, con lo que midió la muñeca: lo que pinta el visor."""

    box: Box
    spec: PackageSpec


@dataclass(frozen=True)
class SupportFace:
    """Cara lateral elegida en el frame original del cubo."""

    axis: int
    sign: int
    distance_m: float

    @property
    def label(self) -> str:
        return f"{'+' if self.sign > 0 else '-'}{'XYZ'[self.axis]}"

    @property
    def normal(self) -> np.ndarray:
        normal = np.zeros(3)
        normal[self.axis] = self.sign
        return normal


@dataclass(frozen=True)
class OrientationRecord:
    """Lo medido y lo conseguido para un cubo."""

    package_id: str
    measured: PackageSpec
    support_face: SupportFace
    pallet_position: np.ndarray
    downward_cog_m: float


@dataclass
class OrientationExperiment:
    """Resultado local del experimento; no es una fila de la plataforma."""

    n_objects: int
    records: list[OrientationRecord] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.records) == self.n_objects


class OrientationExperimentError(RuntimeError):
    """La maniobra no pudo completar uno de sus waypoints físicos."""


def nearest_lateral_face(spec: PackageSpec) -> SupportFace:
    """Devuelve la cara ``+/-X`` o ``+/-Y`` más cercana al CoM medido.

    La distancia a una cara es su semidimensión menos el valor absoluto del offset en
    ese eje. Elegir sólo entre X/Y es deliberado: son las componentes observables en la
    lectura de muñeca a plomo.
    """
    half = np.asarray(spec.dims_m, dtype=float) / 2.0
    cog = np.asarray(spec.cog_offset_m, dtype=float)
    if half.shape != (3,) or cog.shape != (3,) or np.any(half <= 0):
        raise ValueError("dimensiones y CoM deben ser vectores 3D válidos")
    if np.any(np.abs(cog[:2]) > half[:2]):
        raise ValueError("el CoM medido cae fuera de la caja")
    distances = half[:2] - np.abs(cog[:2])
    axis = int(np.argmin(distances))
    sign = 1 if cog[axis] >= 0.0 else -1
    return SupportFace(axis=axis, sign=sign, distance_m=float(distances[axis]))


def support_rotation(face: SupportFace, top_outward_world=(0.0, -1.0, 0.0)) -> np.ndarray:
    """Orientación del cubo que deja ``face`` hacia abajo en la mesa.

    La tapa original queda mirando hacia ``top_outward_world``. En esta celda se usa
    ``-Y`` para que el VGP20 quede en el lado de la mesa auxiliar más próximo al robot
    mientras suelta el cubo de costado.
    """
    if face.axis not in (0, 1) or face.sign not in (-1, 1):
        raise ValueError("el experimento sólo puede apoyar una cara lateral")
    normal = face.normal
    local_top = np.array([0.0, 0.0, 1.0])
    world_down = np.array([0.0, 0.0, -1.0])
    world_top = np.asarray(top_outward_world, dtype=float)
    norm = float(np.linalg.norm(world_top))
    if norm < 1e-9 or abs(float(world_top @ world_down)) > 1e-9:
        raise ValueError("la dirección de la tapa tiene que ser horizontal")
    world_top /= norm
    local_third = np.cross(local_top, normal)
    world_third = np.cross(world_top, world_down)
    rotation = (
        np.outer(world_down, normal)
        + np.outer(world_top, local_top)
        + np.outer(world_third, local_third)
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("la orientación de apoyo no es ortonormal")
    return rotation


def build_orientation_boxes(cfg: dict) -> list[Box]:
    """Construye la carga especial, fuera del catálogo de niveles normales."""
    experiment = cfg["com_orientation_experiment"]
    edge = float(experiment["cube_edge_m"])
    mass = float(experiment["mass_kg"])
    friction = tuple(float(value) for value in experiment["friction"])
    boxes: list[Box] = []
    for index, row in enumerate(experiment["boxes"]):
        boxes.append(Box(
            index=index,
            package_id=f"com-cube-{index:02d}",
            type_name=str(row["name"]),
            dims_m=(edge, edge, edge),
            mass_kg=mass,
            cog_offset_m=tuple(float(value) for value in row["cog_offset_m"]),
            friction=friction,
            rgba=tuple(float(value) for value in row["rgba"]),
        ))
    slots = experiment["pallet_slots_xy"]
    if len(slots) != len(boxes):
        raise ValueError("hace falta una posición de palé por cubo del experimento")
    return boxes


def build_orientation_scene(cfg: dict | None = None, *, seed: int = 1,
                            simplified: bool = False) -> PalletScene:
    """Monta la celda especial sin añadir un nivel seleccionable al paletizado."""
    cfg = load_configs() if cfg is None else cfg
    boxes = build_orientation_boxes(cfg)
    level = Level(
        id=0,
        source="table",
        name="experimento · orientar el CoM hacia el apoyo",
        n_packages=len(boxes),
        types=tuple(box.type_name for box in boxes),
    )
    scene = PalletScene(cfg, level, boxes, seed, simplified=simplified)
    scene.settle(0.20)
    return scene


def run_orientation_experiment(scene: PalletScene, detector: Detector, gauge: Gauge,
                               *, speed: float = 0.0,
                               log: Callable[[str], None] | None = None,
                               ) -> OrientationExperiment:
    """Mide, reorienta en la auxiliar y paletiza todos los cubos."""
    arm = ArmController(scene)
    arm.fast_forward = speed <= 0.0
    scene.speed = max(0.0, float(speed))
    supply = _StationSupply(scene)
    scene.supply = supply
    supply.stage(scene)
    if hasattr(gauge, "calibrate"):
        gauge.calibrate(scene, arm)

    experiment = scene.cfg["com_orientation_experiment"]
    slots = experiment["pallet_slots_xy"]
    result = OrientationExperiment(n_objects=len(scene.boxes))
    # Lo mismo que anota `src/episode.py` para los marcadores de CoM del visor —ver
    # `render.draw_overlay`—: lo que midió la muñeca de cada cubo y los que ya están en
    # el palé. El CoM calculado viaja con el cubo al tumbarlo, porque va en su marco.
    scene.weighed_specs = {}
    scene.load_placements = []
    try:
        for index, box in enumerate(scene.boxes):
            package_id = supply.present(scene)
            if package_id != box.package_id:
                raise OrientationExperimentError(
                    f"la mesa presentó {package_id!r}, se esperaba {box.package_id!r}"
                )
            observations = detector.observe(scene)
            observation = next(
                (item for item in observations if item.package_id == package_id), None
            )
            if observation is None:
                raise OrientationExperimentError(f"no se detectó {package_id}")

            _pick_from_source(arm, scene, box, observation)
            spec = gauge.measure(scene, arm, observation)
            scene.weighed_specs[box.index] = spec
            face = nearest_lateral_face(spec)
            desired_rotation = support_rotation(
                face, experiment.get("top_outward_world", (0.0, -1.0, 0.0))
            )
            _lay_on_auxiliary(scene, arm, box, face, desired_rotation)
            _regrasp_from_above(scene, arm, box, face)
            supply.release(scene)
            _place_on_pallet(scene, arm, box, slots[index])
            scene.load_placements.append(_OnPallet(box, spec))

            position, quaternion = scene.box_pose(box.index)
            actual_rotation = _quat_to_matrix(scene, quaternion)
            downward = -float((actual_rotation @ spec.cog_offset_m)[2])
            record = OrientationRecord(
                package_id=box.package_id,
                measured=spec,
                support_face=face,
                pallet_position=position,
                downward_cog_m=downward,
            )
            result.records.append(record)
            if log is not None:
                dx, dy = (float(value) * 1000 for value in spec.cog_offset_m[:2])
                log(
                    f"{box.package_id}: CoM ({dx:+.1f}, {dy:+.1f}) mm · "
                    f"cara {face.label} abajo · palé {index + 1}/{len(scene.boxes)}"
                )
        arm.go_home()
    except Exception:
        if arm.is_holding():
            arm.release()
        raise
    result.duration_s = round(float(scene.clock), 2)
    return result


def _pick_from_source(arm: ArmController, scene: PalletScene, box: Box,
                      observation: Observation) -> None:
    motion = scene.cfg["motion"]
    top = np.asarray(observation.position, dtype=float).copy()
    top[2] += observation.dims_guess[2] / 2.0
    seal_z = float(top[2]) + arm.cup_gap
    safe_z = max(float(motion["transit_height"]),
                 seal_z + float(motion["place_clearance"]))
    grasp = arm.plan_grasp(box)
    if not arm.go_to(float(top[0]), float(top[1]), safe_z, observation.yaw):
        raise OrientationExperimentError(f"no se pudo alcanzar {box.package_id}")
    if not arm.go_to(
        float(top[0]), float(top[1]), seal_z, observation.yaw, approach=True
    ) or not arm.seal(box.index, grasp):
        raise OrientationExperimentError(f"no se pudo recoger {box.package_id}")
    if not arm.go_to(float(top[0]), float(top[1]), safe_z, observation.yaw):
        raise OrientationExperimentError(f"no se pudo elevar {box.package_id}")


def _lay_on_auxiliary(scene: PalletScene, arm: ArmController, box: Box,
                      face: SupportFace, body_rotation: np.ndarray) -> None:
    """Tumba el cubo agarrado y lo suelta sobre su cara elegida."""
    if scene.held is None or scene.held.index != box.index:
        raise OrientationExperimentError("el gauge no dejó el cubo rígidamente agarrado")
    experiment = scene.cfg["com_orientation_experiment"]
    auxiliary = scene.cfg["auxiliary_table"]
    x, y = (float(value) for value in auxiliary["center"])
    half = np.asarray(box.dims_m, dtype=float) / 2.0
    vertical_half = float(np.abs(body_rotation[2]) @ half)
    drop = float(scene.cfg["motion"]["drop_clearance"])
    centre_z = float(auxiliary["height"]) + vertical_half + drop
    clearance = float(experiment["reorientation_clearance_m"])

    relative_position, relative_rotation = _held_from_tcp(scene)
    tool_rotation = body_rotation @ relative_rotation.T

    def pose_at(z: float) -> Pose:
        body_position = np.array([x, y, z])
        tcp_position = body_position - tool_rotation @ relative_position
        return Pose(tcp_position, tool_rotation)

    if not arm.move_to(pose_at(centre_z + clearance), compensate_held=False):
        raise OrientationExperimentError(f"{box.package_id}: no alcanza la auxiliar en alto")
    contact_pose = pose_at(centre_z)
    if not arm.move_to(contact_pose, approach=True, compensate_held=False):
        raise OrientationExperimentError(f"{box.package_id}: no alcanza la auxiliar al soltar")
    arm.release()
    scene.settle()

    _position, quaternion = scene.box_pose(box.index)
    rotation = _quat_to_matrix(scene, quaternion)
    alignment = float((rotation @ face.normal) @ np.array([0.0, 0.0, -1.0]))
    if alignment < np.cos(np.radians(3.0)):
        raise OrientationExperimentError(
            f"{box.package_id}: la cara {face.label} no quedó apoyada"
        )

    # La herramienta apunta hacia el cubo; se retira en la dirección contraria antes
    # de volver a ponerse vertical. Así no atraviesa el bulto recién depositado.
    away = -contact_pose.rotation[:, 2]
    current = arm.tcp_pose()
    retreat = Pose(current.position + away * clearance, current.rotation)
    if not arm.move_to(retreat, compensate_held=False):
        raise OrientationExperimentError(f"{box.package_id}: no puede retirarse de la auxiliar")


def _regrasp_from_above(scene: PalletScene, arm: ArmController, box: Box,
                        support_face: SupportFace) -> None:
    """Recoge otra vez el cubo por la cara opuesta a la que quedó apoyada."""
    position, quaternion = scene.box_pose(box.index)
    rotation = _quat_to_matrix(scene, quaternion)
    half = np.asarray(box.dims_m, dtype=float) / 2.0
    top = position - rotation @ support_face.normal * half[support_face.axis]
    seal_z = float(top[2]) + arm.cup_gap
    safe_z = max(
        float(scene.cfg["motion"]["transit_height"]),
        seal_z + float(scene.cfg["motion"]["place_clearance"]),
    )
    if not arm.go_to(float(top[0]), float(top[1]), safe_z, 0.0):
        raise OrientationExperimentError(f"{box.package_id}: no alcanza el segundo agarre")
    if not arm.go_to(float(top[0]), float(top[1]), seal_z, 0.0, approach=True):
        raise OrientationExperimentError(f"{box.package_id}: no baja al segundo agarre")
    if not arm.seal(box.index):
        raise OrientationExperimentError(f"{box.package_id}: falla el segundo agarre")
    if not arm.go_to(float(top[0]), float(top[1]), safe_z, 0.0):
        raise OrientationExperimentError(f"{box.package_id}: no eleva el segundo agarre")


def _place_on_pallet(scene: PalletScene, arm: ArmController, box: Box, slot_xy) -> None:
    """Deja el cubo reorientado en una posición fija de la primera capa."""
    x, y = (float(value) for value in slot_xy)
    motion = scene.cfg["motion"]
    target_z = (
        scene.deck_z + box.dims_m[2] + arm.cup_gap
        + float(motion["drop_clearance"])
    )
    safe_z = max(float(motion["transit_height"]),
                 target_z + float(motion["place_clearance"]))
    current = arm.tcp_pose().position
    if not arm.go_to(float(current[0]), float(current[1]), safe_z, 0.0):
        raise OrientationExperimentError(f"{box.package_id}: no sube a tránsito")
    if not arm.go_to(x, y, safe_z, 0.0):
        raise OrientationExperimentError(f"{box.package_id}: no alcanza su hueco del palé")
    if not arm.go_to(x, y, target_z, 0.0, approach=True):
        raise OrientationExperimentError(f"{box.package_id}: no baja a su hueco del palé")
    arm.release()
    scene.settle()


def _held_from_tcp(scene: PalletScene) -> tuple[np.ndarray, np.ndarray]:
    """Pose del centro del cubo respecto al TCP, no respecto a la brida."""
    held = scene.held
    if held is None:
        raise OrientationExperimentError("no hay ningún cubo agarrado")
    relative_rotation = _quat_to_matrix(scene, held.quaternion)
    tcp_in_tool = np.asarray(scene.model.site_pos[scene.tcp_site], dtype=float)
    relative_position = np.asarray(held.position, dtype=float) - tcp_in_tool
    return relative_position, relative_rotation


def _quat_to_matrix(scene: PalletScene, quaternion) -> np.ndarray:
    matrix = np.empty(9)
    scene.mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)
