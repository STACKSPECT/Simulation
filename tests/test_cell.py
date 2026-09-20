"""Pruebas físicas de la celda. Arrancan MuJoCo, pero no usan red."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.cell import tcp_frame  # noqa: E402
from src.cell.arm import ArmController  # noqa: E402
from src.cell.conveyor import Belt, make_supply  # noqa: E402
from src.cell.render import VIEWS  # noqa: E402
from src.cell.scene import build_catalogue, build_scene, levels, load_configs  # noqa: E402
from src.cell.stability import run_stability_test  # noqa: E402
from src.cell.truck import TruckSupply  # noqa: E402
from src.episode import run_episode  # noqa: E402
from src.planner.heuristic import ScorePlanner  # noqa: E402
from src.vision.gauge import WristGauge  # noqa: E402
from src.vision.oracle import OracleDetector, OracleGauge  # noqa: E402


def test_levels_declare_distinct_loads() -> None:
    """Mesa, cinta y camión no son el mismo escenario con otro título."""
    cfg = load_configs()
    catalogue = levels(cfg)
    assert catalogue[11].source == "table"
    assert catalogue[21].source == "conveyor"
    assert catalogue[31].source == "truck"
    load_11 = build_catalogue(cfg, catalogue[11], seed=1)
    load_13 = build_catalogue(cfg, catalogue[13], seed=1)
    load_31 = build_catalogue(cfg, catalogue[31], seed=1)
    assert len(load_11) == 4
    assert len(load_13) == 8
    assert len(load_31) == 5
    assert {box.type_name for box in load_11} == {"std_m"}
    assert {box.type_name for box in load_13} == {"random"}
    assert catalogue[11].cog == "centred"
    assert catalogue[13].cog == "adversarial"
    assert catalogue[13].pos_jitter_m > catalogue[11].pos_jitter_m
    assert catalogue[13].yaw_jitter_deg > catalogue[11].yaw_jitter_deg


def test_scenes_compile_for_all_sources() -> None:
    # El 15 y el 17 van aquí y no en el barrido de IK: son los que estrenan tipos de
    # catálogo, y lo que hay que cazar barato es que su MJCF compile y se asiente.
    #
    # El 34 va por lo mismo y por otra razón: es el único con `decor: plant`, y su MJCF
    # trae 122 geoms de una exportación ajena. Ojo, `simplified=True` NO monta el atrezo
    # —el decorado se salta, como la valla—, así que lo que esta prueba ancla es el
    # resto: los materiales de la nave en el `<asset>`, su suelo recortado y su rig de
    # cinco luces. Que el atrezo entero compile lo cubre el barrido de abajo.
    for level in (11, 15, 17, 21, 31, 34):
        scene = build_scene(level_id=level, seed=3, simplified=True)
        try:
            supply = make_supply(scene)
            supply.stage(scene)
            assert scene.model.nbody > 0
            assert scene.level.id == level
        finally:
            scene.close()


def test_auxiliary_table_exists_in_all_levels() -> None:
    """Todos los niveles montan la misma mesa vacía, además de su propia fuente."""
    cfg = load_configs()
    auxiliary = cfg["auxiliary_table"]
    pallet = cfg["pallet"]
    # El aire al palé, sin presuponer de qué lado cae la mesa: hoy está a la IZQUIERDA
    # (ver la cabecera de `configs/scene.yaml`), y el lado es justo lo que se cambió.
    half = float(pallet["dims"][0]) / 2
    gap = max(
        (float(auxiliary["center"][0]) - float(auxiliary["size"][0]))
        - (float(pallet["center"][0]) + half),
        (float(pallet["center"][0]) - half)
        - (float(auxiliary["center"][0]) + float(auxiliary["size"][0])),
    )
    assert gap >= 0.05 - 1e-9

    # El mayor bulto cabe girado: su ancho va en X y su largo en Y.
    generator = cfg["generator"]
    assert 2 * float(auxiliary["size"][0]) >= float(generator["width"][1])
    assert 2 * float(auxiliary["size"][1]) >= float(generator["length"][1])

    for level_id in sorted(levels(cfg)):
        scene = build_scene(level_id=level_id, simplified=True)
        try:
            mujoco = scene.mujoco
            auxiliary = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_GEOM, "auxiliary_table"
            )
            source_table = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_GEOM, "table"
            )
            assert auxiliary >= 0, f"nivel {level_id}: falta la mesa auxiliar"
            assert (source_table >= 0) is (scene.level.source == "table")
        finally:
            scene.close()


def test_all_four_cameras_exist() -> None:
    scene = build_scene(level_id=11, simplified=True)
    try:
        names = {
            scene.mujoco.mj_id2name(scene.model, scene.mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(scene.model.ncam)
        }
        assert set(VIEWS) <= names
    finally:
        scene.close()


def test_stability_test_reuses_the_loaded_pallet_and_restores_it() -> None:
    """El ensayo mueve la pila real y deja la escena en su estado post-paletizado."""
    scene = build_scene(level_id=11, seed=1, simplified=True)
    try:
        box = scene.boxes[0]
        px, py = scene.pallet_center
        scene.place_box(
            box.index,
            (px, py, scene.deck_z + box.dims_m[2] / 2 + 0.002),
        )
        scene.settle(0.4)
        before = scene.box_pose(box.index)[0].copy()
        fast = {
            **scene.cfg["stability_test"],
            "levels_g": [0.15],
            "axes": ["x"],
            "duration": 0.20,
            "settle_seconds": 0.25,
            "hold_seconds": 0.02,
            "rest_seconds": 0.02,
            "beam_settle_seconds": 0.8,
        }

        result = run_stability_test(scene, [box], fast)

        assert result["ran"]
        assert result["shake"]["summary"]["trial_count"] == 1
        assert result["beam"]["summary"]["trial_count"] == 2
        assert np.allclose(scene.box_pose(box.index)[0], before, atol=0.005)
        assert scene.data.eq_active[scene.pallet_weld]
    finally:
        scene.close()


def test_level_11_stack_survives_all_fifteen_jolts() -> None:
    """Ningún mueble de la celda estorba a la pila mientras se la sacude.

    Es una regresión de MOBILIARIO, no del planificador. El ensayo de estabilidad (#28)
    y la mesa auxiliar (#29) entraron por separado y cada uno estaba bien; juntos no,
    porque la mesa estaba a 50 mm del canto del palé y durante la sacudida el palé se
    suelta de su weld y VIAJA: +309 mm en la de 0.80 g en X. La caja de la capa 2 daba
    contra la mesa y la pila deslizaba 31,5 mm contra un umbral de 30, sin que se cayera
    ninguna caja — o sea, sin ninguna señal que mirase nadie.

    Con la mesa donde está hoy el nivel 11 semilla 1 da 99.9/100, mínima 99.0 y aguanta
    las 15. Si alguien vuelve a acercar un mueble al palé, esto es lo que se entera.
    Las cifras y el barrido, en la cabecera de `configs/scene.yaml: auxiliary_table`.

    `speed=0.0` NO es un detalle: es lo que pone `arm.fast_forward` (ver
    `src/episode.py`) y es el defecto de `scripts/palletize.py` sin visor, o sea la
    configuración en la que se midió todo esto. Con `speed=1.0` el brazo recorre los
    waypoints en vez de teletransportarse, la pila queda distinta y el fallo NO
    aparece ni con la mesa en su sitio viejo: el test pasaría sin vigilar nada.
    """
    from src import measure

    scene = build_scene(level_id=11, seed=1, simplified=True)
    try:
        episode = run_episode(
            scene, OracleDetector(), OracleGauge(), ScorePlanner(scene.cfg),
            seed=1, speed=0.0,
        )
        assert episode.success, episode.failure

        physical = measure.on_pallet(scene, episode.final_placements)
        result = run_stability_test(scene, [item.box for item in physical])

        assert result["ran"]
        shake = result["shake"]["summary"]
        assert shake["trial_count"] == 15
        assert shake["held_all"], (
            "alguna sacudida movió la pila más de la cuenta; si acabas de mover un "
            f"mueble, ése es el primer sospechoso. min_score={shake['min_score']}"
        )
        assert result["beam"]["summary"]["held_all"]
    finally:
        scene.close()


def _shake_contacts_with(level: int, seed: int, geom_name: str) -> tuple[int, dict]:
    """Contactos entre la pila y un geom nombrado DURANTE las 15 sacudidas.

    Cuenta sólo el transporte: la viga tumba la pila a propósito en los niveles
    profundos y ahí sí acaban cayendo cajas sobre cualquier cosa que haya al lado.
    """
    from src import measure
    from src.cell import stability as stability_module

    scene = build_scene(level_id=level, seed=seed, simplified=True)
    original = scene.mujoco.mj_step
    try:
        episode = run_episode(
            scene, OracleDetector(), OracleGauge(), ScorePlanner(scene.cfg),
            seed=seed, speed=0.0,
        )
        boxes = [item.box for item in measure.on_pallet(scene, episode.final_placements)]
        assert boxes, f"nivel {level} semilla {seed}: no quedó ninguna caja en el palé"
        target = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        assert target >= 0, f"no existe el geom {geom_name!r}"
        tally = 0

        def counting_step(model, data, nstep=1):
            nonlocal tally
            original(model, data, nstep)
            for index in range(data.ncon):
                contact = data.contact[index]
                if target in (int(contact.geom1), int(contact.geom2)):
                    tally += 1

        scene.mujoco.mj_step = counting_step
        trial = stability_module._StabilityTrial(
            scene, boxes, dict(scene.cfg["stability_test"])
        )
        # Las sacudidas PRIMERO y el return después: en `return tally, run(...)` Python
        # evalúa `tally` antes de correr nada y siempre devolvería cero.
        summary = trial.run_transport()["summary"]
        return tally, summary
    finally:
        scene.mujoco.mj_step = original
        scene.close()


def test_no_furniture_touches_the_stack_while_it_is_shaken() -> None:
    """El guardia bueno del mobiliario: CERO contactos, no un umbral de puntuación.

    Mirar sólo la puntuación no vale. El nivel 13 semilla 1 es el caso justo: su pila
    asoma 48,8 mm por el canto izquierdo del palé YA EN REPOSO —el vuelo lo pone el
    planificador, no la sacudida— así que en planta pasa a 1,1 mm de la mesa, y aun así
    aguanta las 15. Un test de puntuación lo daría por bueno estando a un pelo.

    Lo que de verdad mantiene la mesa fuera del ensayo es que su único geom con
    colisión es la tapa, z[0.50, 0.58], y las cajas voladas cuelgan a cotas de capa que
    no cruzan esa banda. Eso NO se ve en la puntuación: se ve contando contactos. Si
    alguien sube `height`, engorda la tapa, le da colisión a las patas o vuelve a
    arrimar un mueble al palé, esto se entera y la puntuación puede no enterarse.

    Medido: cero contactos en 24 celdas (niveles 11/13/21/23/31/33 x semillas 1-4), y
    los tres números del resumen idénticos a correr con la mesa ausente. La tabla está
    en la cabecera de `configs/scene.yaml: auxiliary_table`.
    """
    for level, seed in ((13, 1), (11, 1)):
        contacts, summary = _shake_contacts_with(level, seed, "auxiliary_table")
        assert contacts == 0, (
            f"nivel {level} semilla {seed}: la mesa auxiliar tocó la pila en "
            f"{contacts} pasos de las sacudidas. Contaminar el ensayo no se ve en la "
            f"puntuación (salió {summary['mean_score']}/100, mínima "
            f"{summary['min_score']}): mira dónde está el mueble."
        )


def test_belt_moves_the_package_through_physics() -> None:
    scene = build_scene(level_id=21, seed=4, simplified=True)
    try:
        supply = Belt(scene)
        supply.stage(scene)
        positions: list[float] = []
        original_step = scene.step

        def observe(seconds: float) -> None:
            original_step(seconds)
            if supply.current is not None:
                positions.append(float(scene.box_pose(supply.current)[0][0]))

        scene.step = observe
        package_id = supply.present(scene)
        assert package_id is not None
        assert len({round(value, 3) for value in positions}) > 10
        assert positions[-1] - positions[0] > 0.4
        assert abs(positions[-1] - scene.cfg["conveyor"]["station"][0]) < 0.04
    finally:
        scene.close()


def _assert_delivered_at_rest(scene, supply, *, window_s: float = 0.5) -> None:
    """Tras `present()`, el centro de la tapa no se mueve en la ventana de aproximación."""
    assert supply.current is not None
    top = scene.box_top_center(supply.current).copy()
    scene.settle(window_s)
    moved = scene.box_top_center(supply.current) - top
    drift_mm = float(np.linalg.norm(moved) * 1000)
    assert drift_mm < 5.0, f"deriva {drift_mm:.1f} mm del centro de la tapa en {window_s:.1f} s"


def test_belt_present_waits_until_level_22_seed_1_is_still() -> None:
    """Nivel 22, semilla 1: devolver con el cartón rodando era un offset de 20 mm."""
    scene = build_scene(level_id=22, seed=1, simplified=True)
    try:
        supply = Belt(scene)
        supply.stage(scene)
        assert supply.present(scene) is not None
        _assert_delivered_at_rest(scene, supply, window_s=1.0)
        assert float(scene.cfg["episode"]["tolerance_xy"]) == 0.020
    finally:
        scene.close()


def test_belt_present_settles_jittered_deliveries() -> None:
    """Llegadas descentradas y giradas también tienen que quedar quietas.

    El brazo se lleva el cartón antes de `release()`; sin apartarlo, el siguiente
    choca con el que sigue en la estación y la espera expira.
    """
    for seed in (1, 2, 7):
        scene = build_scene(level_id=22, seed=seed, simplified=True)
        try:
            supply = Belt(scene)
            supply.stage(scene)
            for _ in range(3):
                assert supply.present(scene) is not None, f"semilla {seed}: sin entrega"
                _assert_delivered_at_rest(scene, supply)
                scene.park_box(supply.current)
                supply.release(scene)
        finally:
            scene.close()


def test_the_belt_delivers_every_package_fully_supported() -> None:
    """Entregar es dejar el cartón ENTERO sobre la banda, plano y quieto.

    Los tres niveles fallaban con la estación en el canto de la banda: el bulto paraba
    con medio cuerpo en el aire, volcaba y se iba hacia atrás —o al suelo— mientras el
    brazo viajaba. Se mira la huella GIRADA, no el centro: un cartón de 0.55 m a 30°
    ocupa 0.33 m de semihuella en X y es justo el que se salía.
    """
    for level in (21, 22, 23):
        scene = build_scene(level_id=level, seed=1, simplified=True)
        try:
            belt = Belt(scene)
            belt.stage(scene)
            x, y = scene.cfg["conveyor"]["center"]
            length, width = scene.cfg["conveyor"]["dims"]
            for _ in range(len(scene.boxes)):
                package_id = belt.present(scene)
                assert package_id is not None, f"nivel {level}: la cinta no entregó"
                index = belt.current
                body = scene.body_id(index)
                cx, cy, _ = scene.data.xpos[body]
                rotation = scene.data.xmat[body].reshape(3, 3)
                half_x, half_y, _ = (value / 2 for value in scene.boxes[index].dims_m)
                # Semihuella del rectángulo girado, que es lo que apoya de verdad.
                reach_x = abs(rotation[0, 0]) * half_x + abs(rotation[0, 1]) * half_y
                reach_y = abs(rotation[1, 0]) * half_x + abs(rotation[1, 1]) * half_y
                assert cx - reach_x >= x - length / 2 and cx + reach_x <= x + length / 2
                assert cy - reach_y >= y - width / 2 and cy + reach_y <= y + width / 2
                assert rotation[2, 2] > math.cos(math.radians(2.0))
                belt.release(scene)
                scene.park_box(index)
        finally:
            scene.close()


def test_belt_times_out_if_the_package_never_settles() -> None:
    """Si llega y no se asienta, es `timeout`, no una entrega con pose caducada."""
    scene = build_scene(level_id=21, seed=1, simplified=True)
    try:
        supply = Belt(scene)
        supply.timeout_s = 8.0
        supply._resting = lambda _scene, _index, previous, _dt: (False, previous)
        supply.stage(scene)
        assert supply.present(scene) is None
        assert supply.jammed
        assert supply.current is None
    finally:
        scene.close()


class _CountingViewer:
    """Un visor de mentira: sólo cuenta cuántas veces le piden refrescar."""

    def __init__(self) -> None:
        self.syncs = 0

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.syncs += 1


def test_the_viewer_sets_the_pace_and_does_not_sync_every_step() -> None:
    """`speed` es el ritmo, y refrescar 500 veces por segundo no es más fluido.

    Antes `--speed` sólo decidía si el brazo teletransporta: cualquier valor positivo
    corría igual. Y se sincronizaba una vez por paso de 2 ms, ocho veces la pantalla.
    """
    scene = build_scene(level_id=21, seed=1, simplified=True)
    try:
        scene.viewer = _CountingViewer()
        scene.speed = 2.0                     # 2 s simulados por segundo real
        start = time.monotonic()
        scene.step(0.5)                       # 250 pasos: 0.25 s de reloj de pared
        elapsed = time.monotonic() - start
        # Sin acompasar, estos 250 pasos se van en ~13 ms: el ritmo es lo que los frena.
        assert 0.20 < elapsed < 1.0, elapsed
        # Y a 60 Hz eso son ~15 refrescos, no 250.
        assert 1 <= scene.viewer.syncs < 100, scene.viewer.syncs
    finally:
        scene.viewer = None
        scene.close()


def test_truck_always_presents_the_highest_box() -> None:
    scene = build_scene(level_id=31, seed=5, simplified=True)
    try:
        supply = TruckSupply(scene)
        supply.stage(scene)
        while supply.pending:
            highest = max(
                supply.pending,
                key=lambda index: (float(scene.box_top_center(index)[2]), -index),
            )
            assert supply.present(scene) == scene.boxes[highest].package_id
            supply.release(scene)
    finally:
        scene.close()


def test_ik_reaches_the_pick_and_pallet_envelope() -> None:
    for level in (11, 21, 31):
        scene = build_scene(level_id=level, seed=6, simplified=True)
        try:
            supply = make_supply(scene)
            supply.stage(scene)
            arm = ArmController(scene)
            targets = []
            if scene.level.source == "conveyor":
                station_x, station_y = scene.station
                for box in scene.boxes:
                    targets.append((station_x, station_y,
                                    scene.surface_z + box.dims_m[2] + arm.cup_gap, 0.0))
            else:
                # Sólo los bultos que están PUESTOS. La mesa no da para todos —ocho
                # sorteados suman más área que ella— y los que no caben esperan
                # aparcados fuera de la escena, a x = -3 y más allá: pedirle al brazo que
                # llegue hasta ahí no prueba nada sobre su alcance. Entran a la mesa
                # cuando queda hueco, y entonces sí caen dentro de esta envolvente.
                staged = getattr(supply, "staged", -1)
                for box in scene.boxes:
                    if staged != -1 and box.index != staged:
                        continue
                    top = scene.box_top_center(box.index)
                    targets.append((float(top[0]), float(top[1]),
                                    float(top[2]) + arm.cup_gap, scene.box_yaw(box.index)))
            px, py = scene.pallet_center
            dx, dy = scene.pallet_dims
            for x in (px - dx * 0.32, px, px + dx * 0.32):
                for y in (py - dy * 0.30, py, py + dy * 0.30):
                    for z in (scene.deck_z + 0.10, scene.deck_z + 0.55):
                        targets.append((x, y, z, 0.0))
            # La mesa auxiliar sólo es útil si se puede depositar el bulto más alto y
            # hacer también el waypoint de aproximación sobre el centro de su superficie.
            auxiliary = scene.cfg["auxiliary_table"]
            ax, ay = (float(value) for value in auxiliary["center"])
            tallest = float(scene.cfg["generator"]["height"][1])
            drop_z = (
                float(auxiliary["height"]) + tallest + arm.cup_gap
                + float(scene.cfg["motion"]["drop_clearance"])
            )
            targets.extend([
                (ax, ay, drop_z, np.pi / 2),
                (ax, ay, drop_z + float(scene.cfg["motion"]["place_clearance"]), np.pi / 2),
            ])
            failed = [
                target for target in targets
                if arm.solve_ik(target[:3], tcp_frame(target[:3], target[3]).rotation) is None
            ]
            assert not failed, f"nivel {level}: poses fuera de alcance: {failed}"
        finally:
            scene.close()


def test_joint_targets_use_the_nearest_equivalent_angle() -> None:
    """Una consigna al otro lado de ±π no debe ordenar casi una vuelta completa."""
    scene = build_scene(level_id=11, simplified=True)
    try:
        arm = ArmController(scene)
        reference = np.asarray(scene.data.qpos[scene.arm_qpos], dtype=float).copy()
        reference[0] = 3.10
        target = reference.copy()
        target[0] = -3.10
        nearest = arm._nearest_joint_target(target, reference)
        assert abs(nearest[0] - reference[0]) < 0.10
        low, high = arm._joint_ranges()[0]
        assert low <= nearest[0] <= high
    finally:
        scene.close()


def test_level_11_physical_cycle_finishes_under_50_simulated_seconds() -> None:
    """Regresión del ciclo que tardaba 75,9 s por parar después de cada waypoint."""
    scene = build_scene(level_id=11, seed=1, simplified=True)
    try:
        episode = run_episode(
            scene,
            OracleDetector(),
            OracleGauge(),
            ScorePlanner(scene.cfg),
            seed=1,
            speed=1.0,
        )
        assert episode.success, episode.failure
        assert episode.duration_s < 50.0
    finally:
        scene.close()


def test_robot_can_leave_a_package_on_the_auxiliary_table() -> None:
    """La mesa no sólo existe: sostiene un bulto depositado por el brazo."""
    scene = build_scene(level_id=11, seed=3, simplified=True)
    arm = ArmController(scene)
    arm.fast_forward = True
    try:
        supply = make_supply(scene)
        supply.stage(scene)
        assert supply.present(scene) is not None
        assert supply.current is not None
        box = scene.boxes[supply.current]

        top = scene.box_top_center(box.index)
        yaw = scene.box_yaw(box.index)
        seal_z = float(top[2]) + arm.cup_gap
        safe_z = max(
            float(scene.cfg["motion"]["transit_height"]),
            seal_z + float(scene.cfg["motion"]["place_clearance"]),
        )
        assert arm.go_to(float(top[0]), float(top[1]), safe_z, yaw)
        assert arm.go_to(float(top[0]), float(top[1]), seal_z, yaw, approach=True)
        assert arm.seal(box.index)
        assert arm.go_to(float(top[0]), float(top[1]), safe_z, yaw)

        auxiliary = scene.cfg["auxiliary_table"]
        x, y = (float(value) for value in auxiliary["center"])
        yaw = np.pi / 2                         # el bulto largo cabe girado en la mesa
        target_z = (
            float(auxiliary["height"]) + box.dims_m[2] + arm.cup_gap
            + float(scene.cfg["motion"]["drop_clearance"])
        )
        safe_z = target_z + float(scene.cfg["motion"]["place_clearance"])
        assert arm.go_to(x, y, safe_z, yaw)
        assert arm.go_to(x, y, target_z, yaw, approach=True)
        arm.release()
        scene.settle()

        position, _ = scene.box_pose(box.index)
        bottom = float(position[2] - box.dims_m[2] / 2)
        assert np.linalg.norm(position[:2] - np.array([x, y])) < 0.01
        assert abs(bottom - float(auxiliary["height"])) < 0.005
    finally:
        if arm.held is not None:
            arm.release()
        scene.close()


def test_wrist_gauge_recovers_mass_and_planar_cog() -> None:
    """Una lectura a plomo estima masa y CoG en planta; la altura se ancla al centro."""
    scene = build_scene(level_id=11, seed=3, simplified=True)
    arm = ArmController(scene)
    arm.fast_forward = True
    try:
        supply = make_supply(scene)
        supply.stage(scene)
        scene.supply = supply
        gauge = WristGauge()
        gauge.calibrate(scene, arm)
        package_id = supply.present(scene)
        assert package_id is not None
        observation = OracleDetector().observe(scene)[0]
        box = next(item for item in scene.boxes if item.package_id == package_id)
        top = observation.position.copy()
        top[2] += observation.dims_guess[2] / 2
        seal_z = float(top[2]) + arm.cup_gap
        safe_z = seal_z + 0.18
        assert arm.go_to(float(top[0]), float(top[1]), safe_z, observation.yaw)
        assert arm.go_to(float(top[0]), float(top[1]), seal_z, observation.yaw, approach=True)
        assert arm.seal(box.index)
        assert arm.go_to(float(top[0]), float(top[1]), safe_z, observation.yaw)
        spec = gauge.measure(scene, arm, observation)
        assert abs(spec.mass_kg - box.mass_kg) < 0.05
        assert abs(spec.cog_offset_m[0] - box.cog_offset_m[0]) < 0.003
        assert abs(spec.cog_offset_m[1] - box.cog_offset_m[1]) < 0.003
        assert abs(spec.cog_offset_m[2]) < 0.003
        weld = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{box.index}"
        )
        assert weld < 0
        assert scene.held is not None and scene.held.index == box.index
    finally:
        if arm.held is not None:
            arm.release()
        scene.close()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} comprobaciones físicas pasadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
