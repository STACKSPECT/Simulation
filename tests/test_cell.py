"""Pruebas físicas de la celda. Arrancan MuJoCo, pero no usan red."""

from __future__ import annotations

import itertools
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.cell import tcp_frame  # noqa: E402
from src.cell.arm import ArmController  # noqa: E402
from src.cell.conveyor import Belt, make_supply  # noqa: E402
from src.cell.render import (  # noqa: E402
    ERROR_RGBA,
    ESTIMATED_COM_RGBA,
    ESTIMATED_LOAD_RGBA,
    GHOST_ALPHA,
    LOAD_RADIUS,
    PACKAGE_RADIUS,
    TRUE_COM_RGBA,
    TRUE_LOAD_RGBA,
    VIEWS,
    draw_overlay,
)
from src.cell.scene import (  # noqa: E402
    PACKAGE_GROUP,
    build_catalogue,
    build_scene,
    levels,
    load_configs,
)
from src.cell.stability import run_stability_test  # noqa: E402
from src.cell.truck import TruckSupply  # noqa: E402
from src.episode import run_episode  # noqa: E402
from src.planner.heuristic import ScorePlanner  # noqa: E402
from src.training import simulate_placement  # noqa: E402
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
    pallet_right = float(pallet["center"][0]) + float(pallet["dims"][0]) / 2
    table_left = float(auxiliary["center"][0]) - float(auxiliary["size"][0])
    assert table_left - pallet_right >= 0.05 - 1e-9

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


def test_the_1_key_hides_the_packages_and_nothing_else() -> None:
    """El grupo de geoms de los bultos es el atajo del visor que los esconde.

    Tiene que quedarse con los bultos enteros —cartón, cinta y etiqueta— y con nada más:
    si otro geom cae en él, el `1` lo esconde también. Y lo que las cámaras ven sale de
    la `MjvOption` por defecto: si el grupo deja de estar encendido ahí, las fotos y el
    mapa de alturas medido se quedan sin bultos sin dar error.
    """
    import mujoco

    assert mujoco.MjvOption().geomgroup[PACKAGE_GROUP] == 1
    # El 34 es el de la nave; sin gráficos simples es cuando monta su atrezo.
    for level_id, simplified in itertools.product((11, 21, 31, 34), (True, False)):
        scene = build_scene(level_id=level_id, seed=3, simplified=simplified)
        try:
            model = scene.model
            packages = {box.body for box in scene.boxes}
            for geom in range(model.ngeom):
                body = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]
                )
                in_group = int(model.geom_group[geom]) == PACKAGE_GROUP
                assert in_group is (body in packages), (
                    f"nivel {level_id}: el geom {geom} de {body} en el grupo "
                    f"{model.geom_group[geom]}"
                )
        finally:
            scene.close()


def test_the_com_markers_show_through_translucent_boxes() -> None:
    """Las casillas de centro de masa pintan algo, y donde dicen.

    Antes las dos acababan en `--show-com`, que sólo añade el CoG al informe: se
    encendían y el visor no enseñaba nada. Y un CoG está DENTRO de su cartón, así que
    además hay que ver a través: con el grupo de los bultos apagado, `draw_overlay` los
    pinta translúcidos sin tocar el `rgba` del modelo, que leen también las cámaras.
    """
    import mujoco

    scene = build_scene(level_id=11, seed=1, simplified=False)
    try:
        # El 0 está en el palé; el 1 ya se pesó pero no se ha soltado —como si
        # estuviera en la mano—, y los demás siguen sin tocar.
        box, weighed = scene.boxes[0], scene.boxes[1]
        px, py = scene.pallet_center
        scene.place_box(box.index, (px, py, scene.deck_z + box.dims_m[2] / 2 + 0.002))
        scene.settle(0.5)
        # Lo que cree la celda, desviado a propósito para que el error no sea cero.
        believed = np.asarray(box.cog_offset_m, dtype=float) + np.array([0.03, -0.02, 0.0])
        spec = SimpleNamespace(cog_offset_m=believed, mass_kg=box.mass_kg)
        in_hand = SimpleNamespace(cog_offset_m=np.array([-0.01, 0.02, 0.0]),
                                  mass_kg=weighed.mass_kg)
        scene.weighed_specs = {box.index: spec, weighed.index: in_hand}
        scene.load_placements = [SimpleNamespace(box=box, spec=spec)]
        opt = mujoco.MjvOption()
        scene.viewer = SimpleNamespace(
            user_scn=mujoco.MjvScene(scene.model, maxgeom=2000), opt=opt,
        )
        alphas_before = scene.model.geom_rgba[:, 3].copy()

        scene.heightmap_geoms = 5                 # la rejilla va delante y no se pisa
        scene.viewer.user_scn.ngeom = 5
        draw_overlay(scene)
        assert scene.viewer.user_scn.ngeom == 5, "sin casillas no se pinta nada"

        scene.show_true_com = scene.show_estimated_com = True
        opt.geomgroup[PACKAGE_GROUP] = 0          # como lo deja `palletize.py`
        draw_overlay(scene)
        scn = scene.viewer.user_scn
        geoms = [scn.geoms[i] for i in range(5, scn.ngeom)]
        ghosts = [g for g in geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX]
        assert len(ghosts) == len(scene.boxes)
        assert all(g.rgba[3] <= GHOST_ALPHA + 1e-6 for g in ghosts)
        assert np.array_equal(scene.model.geom_rgba[:, 3], alphas_before)

        def belief_of(index, offset):
            body = scene.body_id(index)
            rotation = scene.data.xmat[body].reshape(3, 3)
            return scene.data.xpos[body] + rotation @ offset

        def spheres(rgba, radius):
            return [
                np.array(g.pos) for g in geoms
                if g.type == mujoco.mjtGeom.mjGEOM_SPHERE
                and np.allclose(g.rgba, rgba, atol=1e-3)
                and np.isclose(g.size[0], radius)
            ]

        def matches(points, expected):
            # `mjvGeom.pos` es float32: se compara con tolerancia, no redondeando.
            left = [np.asarray(point, dtype=float) for point in expected]
            if len(points) != len(left):
                return False
            for point in points:
                hit = next((i for i, other in enumerate(left)
                            if np.allclose(point, other, atol=1e-5)), None)
                if hit is None:
                    return False
                left.pop(hit)
            return True

        # Verde en TODOS los bultos, estén donde estén; naranja sólo en los pesados.
        assert matches(spheres(TRUE_COM_RGBA, PACKAGE_RADIUS),
                       [scene.data.xipos[scene.body_id(b.index)] for b in scene.boxes])
        assert matches(spheres(ESTIMATED_COM_RGBA, PACKAGE_RADIUS),
                       [belief_of(box.index, believed),
                        belief_of(weighed.index, in_hand.cog_offset_m)])

        # La pila es sólo lo que está en el palé: con un bulto, su mismo punto, pero
        # en sus propios colores, y sin texto en ningún geom.
        truth = scene.data.xipos[scene.body_id(box.index)]
        belief = belief_of(box.index, believed)
        for rgba, point in ((TRUE_LOAD_RGBA, truth), (ESTIMATED_LOAD_RGBA, belief)):
            assert matches(spheres(rgba, LOAD_RADIUS), [point])
            # La plomada acaba en la cubierta, justo debajo.
            feet = spheres(rgba, PACKAGE_RADIUS * 0.8)
            assert len(feet) == 1
            assert np.allclose(feet[0][:2], point[:2], atol=1e-3)
            assert abs(feet[0][2] - scene.deck_z) < 2e-3
        assert all(g.label == "" for g in geoms)
        errors = [g for g in geoms if np.allclose(g.rgba, ERROR_RGBA, atol=1e-3)]
        assert len(errors) == 1
        assert np.allclose(np.array(errors[0].pos), (truth + belief) / 2, atol=1e-6)

        opt.geomgroup[PACKAGE_GROUP] = 1          # la tecla `1`: bultos opacos
        draw_overlay(scene)
        geoms = [scn.geoms[i] for i in range(5, scn.ngeom)]
        assert not [g for g in geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX]
    finally:
        scene.viewer = None
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


def test_weight_training_places_and_tests_without_the_robot() -> None:
    """El banco salta brazo y fuente, pero conserva caída, contacto y estabilidad."""
    scene = build_scene(level_id=11, seed=1, simplified=True, with_robot=False)
    try:
        robot = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, "ur10e_base"
        )
        assert robot < 0
        assert scene.model.nu == 0
    finally:
        scene.close()

    fast = {
        "levels_g": [0.15],
        "axes": ["x"],
        "duration": 0.20,
        "settle_seconds": 0.25,
        "hold_seconds": 0.02,
        "rest_seconds": 0.02,
        "beam_settle_seconds": 0.8,
    }
    progress = []
    result = simulate_placement(
        level=11, seed=1, stability_config=fast, progress=progress.append,
    )
    assert result.n_planned >= 1
    assert result.n_placed >= 1
    assert result.n_on_pallet >= 1
    assert result.stability_trials == 3
    assert result.falls["opportunities"] == result.n_on_pallet * 3
    assert any(row["phase"] == "placement" for row in progress)
    assert sum(row["phase"] == "stability" for row in progress) == 3


def test_belt_moves_the_package_through_physics() -> None:
    """Carga abajo, sube por contacto y cruza los dos tramos en ambos modos gráficos."""
    for level_id in (11, 21):
        for simplified in (True, False):
            scene = build_scene(level_id=level_id, seed=4, simplified=simplified)
            try:
                supply = make_supply(scene)
                supply.stage(scene)
                model = scene.model
                first, second = (model.geom(name) for name in ("belt_segment_a", "belt_segment_b"))
                seam = float(first.pos[0] + first.size[0])
                assert np.isclose(seam, second.pos[0] - second.size[0])
                assert np.isclose(first.pos[2] + first.size[2], supply.surface_z)
                assert np.isclose(second.pos[2] + second.size[2], supply.surface_z)
                platform = model.geom("elevator_platform")
                assert platform.contype != 0 and platform.conaffinity != 0
                platform_pos = scene.data.geom_xpos[platform.id].copy()
                assert np.isclose(platform_pos[2] + platform.size[2], supply.lower_height)
                assert np.isclose(platform_pos[0] + platform.size[0], first.pos[0] - first.size[0])
                header = model.geom("elevator_header")
                assert header.rgba[3] == 1
                front = model.geom("black_box_4")
                assert front.rgba[3] == 1
                assert np.isclose(front.pos[2] + front.size[2], supply.surface_z)
                assert front.pos[0] > max(scene.box_pose(box.index)[0][0] for box in scene.boxes)
                positions: list[float] = []
                rising: list[tuple[float, float]] = []
                loads = []
                original_step = scene.step
                original_place = scene.place_box

                def load(index, position, yaw=0.0):
                    loads.append((index, position))
                    original_place(index, position, yaw)

                scene.place_box = load

                def observe(seconds: float) -> None:
                    original_step(seconds)
                    if supply.current is not None:
                        pos = scene.box_pose(supply.current)[0]
                        positions.append(float(pos[0]))
                        if abs(pos[0] - supply.entry_x) < 0.02:
                            support_z = scene.data.geom_xpos[platform.id, 2] + platform.size[2]
                            rising.append((float(pos[2]), float(support_z)))

                scene.step = observe
                package_id = supply.present(scene)
                assert package_id is not None
                assert positions[0] < seam < positions[-1]
                assert len({round(value, 3) for value in positions if value < seam}) > 10
                assert len({round(value, 3) for value in positions if value > seam}) > 10
                assert abs(positions[-1] - supply.station_x) < 0.04
                assert len(loads) == 1, "el ascensor no debe reescribir la pose del paquete"
                half_box = scene.boxes[0].dims_m[2] / 2
                assert np.isclose(loads[0][1][2], supply.lower_height + half_box + 0.002)
                heights = np.array(rising)
                assert heights[-1, 0] - heights[0, 0] > 0.50
                assert len(set(np.round(heights[:, 0], 3))) > 100
                assert np.max(np.abs(heights[:, 0] - half_box - heights[:, 1])) < 0.01
                before = scene.data.mocap_pos.copy()
                scene.rebuild(None)
                assert np.allclose(scene.data.mocap_pos, before), "sellar no reinicia el ascensor"
                # El siguiente ciclo empieza con la plataforma arriba y debe volver a cargar abajo.
                supply.release(scene)
                scene.park_box(0)
                loads.clear()
                assert supply.present(scene) is not None
                assert len(loads) == 1
                half_box = scene.boxes[loads[0][0]].dims_m[2] / 2
                assert np.isclose(loads[0][1][2], supply.lower_height + half_box + 0.002)
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
    """El mismo plazo cubre el ascensor y el reposo final de la cinta."""
    for timeout in (0.05, 15.0):
        scene = build_scene(level_id=21, seed=1, simplified=True)
        try:
            supply = Belt(scene)
            supply.timeout_s = timeout
            supply._resting = lambda _scene, _index, previous, _dt: (False, previous)
            supply.stage(scene)
            assert supply.present(scene) is None
            assert supply.jammed
            assert supply.current is None
            if timeout > 1.0:
                assert abs(scene.box_pose(0)[0][0] - supply.station_x) < 0.04
            else:
                assert scene.box_pose(0)[0][2] < supply.surface_z
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
            if scene.level.source in ("conveyor", "table"):
                # Las dos entregan en una ESTACIÓN fija —bajo el brazo la cinta, en el
                # centro de la mesa la banda de mesa— y los que esperan están dentro de
                # la caja negra, a x = -2 y más allá. Pedirle al brazo que llegue hasta
                # ahí no prueba nada sobre su alcance: el bulto sale de ahí en la banda.
                station_x, station_y = scene.station
                for box in scene.boxes:
                    targets.append((station_x, station_y,
                                    scene.surface_z + box.dims_m[2] + arm.cup_gap, 0.0))
            else:
                # El camión presenta la carga entera en la bahía, así que aquí sí hay que
                # poder llegar a todas.
                for box in scene.boxes:
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


def test_level_11_physical_cycle_stays_within_its_time_budget() -> None:
    """Regresión del ciclo que paraba después de cada waypoint.

    El nivel 11 pasó de 70,4 s con banda a 99,0 s con ascensor (semilla 1, speed=1):
    cuatro subidas de 2,7 s, tres bajadas y el recorrido extra antes de la cinta.
    115 s conserva unos 16 s de margen; el antiguo coste de parar en cada waypoint
    añadiría unos 25 s y seguiría superándolo.
    """
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
        assert episode.duration_s < 115.0, episode.duration_s
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
