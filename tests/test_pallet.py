"""Contrato rápido del paletizado. No arranca MuJoCo ni usa la red."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from theker_telemetry import FAILURES, TASKS  # noqa: E402

from placing import METRIC_NAMES  # noqa: E402
from scripts.palletize import (  # noqa: E402
    _aborted,
    _request_unwind,
    install_termination_unwind,
    oracle_for,
    parser,
)
from src import measure  # noqa: E402
from src.cell.render import HEIGHTMAP_BUDGET, VIEWS, draw_heightmap  # noqa: E402
from src.cell.scene import SOURCE_DECADE, SOURCES, Level, levels, load_configs  # noqa: E402
from src.contracts import Heightmap, PackageSpec, PlacementPlan  # noqa: E402
from src.episode import Episode  # noqa: E402
from src.planner.heightmap import _stamp_static_obstacles  # noqa: E402
from src.planner.heuristic import ScorePlanner  # noqa: E402
from src.planner.naive import BeamPlanner, GridPlanner  # noqa: E402
from src.telemetry import pallet_state_row, placement_row, run_config  # noqa: E402
from src.vision.surface import (  # noqa: E402
    CameraIntrinsics, CameraPose, fuse_max, rasterize_points, unproject_depth,
)

CFG = load_configs(REPO)
EVENT_KINDS = {"perceive", "plan", "pick", "place", "settle", "fail"}
EXPECTED_FAILURES = {
    "no_detection", "ik_unreachable", "collision", "grasp_slip",
    "wrong_placement", "timeout", "stack_collapse", "overhang_violation",
}


def _placement(x=0.0, y=0.0, *, placed=True, cog=(0.0, 0.0, 0.0)):
    spec = PackageSpec("box-1", "std_m", (0.42, 0.30, 0.18), 4.2, np.array(cog))
    plan = PlacementPlan(np.array([x, y, 0.09]), 0.0, 1, "slot-1", 0.8, 1.0,
                         {"support": 1.0})
    box = SimpleNamespace(package_id="box-1", type_name="std_m")
    position = np.array([x, y, 0.09])
    return measure.Placement(
        box, spec, plan, position, 0.0, position.copy(), 0.0, 0.0,
        1.0, 0.0, placed,
    )


def test_rows_match_supabase_columns() -> None:
    placement = _placement()
    assert set(placement_row(0, placement)) == {
        "seq", "package_id", "package_type", "mass_kg", "dims_m", "layer",
        "planned_pose", "actual_pose", "error_xy_m", "error_yaw_rad",
        "support_ratio", "overhang_m", "placed",
    }
    state = measure.pallet_state([placement])
    assert set(pallet_state_row(0, state, 0.002)) == {
        "after_seq", "mass_kg", "cog_x", "cog_y", "cog_z",
        "stability_margin_m", "fill_ratio", "settle_drift_m",
    }


def test_event_rows_use_closed_vocabulary_and_unique_seq() -> None:
    episode = Episode(seed=1, n_objects=1)
    for seq, kind in enumerate(("perceive", "pick", "plan", "place", "settle", "fail")):
        episode.events.append({
            "ts": float(seq), "seq": seq, "kind": kind,
            "package_id": "box-1", "payload": {},
        })
    assert {event["kind"] for event in episode.events} <= EVENT_KINDS
    seqs = [event["seq"] for event in episode.events]
    assert seqs == sorted(set(seqs))


def test_views_and_failures_match_the_schema() -> None:
    assert set(CFG["cameras"]) == set(VIEWS) == {"top", "side", "iso", "camera"}
    assert set(FAILURES) == EXPECTED_FAILURES
    # `task` es la fuente. Si el SDK no la conoce, `EpisodeResult` lanza y el episodio
    # no sube: que falle aquí en un segundo y no a mitad de un run.
    assert set(SOURCES) <= set(TASKS), sorted(set(SOURCES) - set(TASKS))


def test_level_ids_encode_the_source() -> None:
    catalogue = levels(CFG)
    assert len(catalogue) == 11
    assert len(set(catalogue)) == 11
    assert {level.source for level in catalogue.values()} == set(SOURCES)
    for level in catalogue.values():
        assert level.id // 10 == SOURCE_DECADE[level.source]


def test_the_stability_protocol_is_explicit_and_opt_in() -> None:
    protocol = CFG["stability_test"]
    assert protocol["levels_g"] == [0.05, 0.15, 0.30, 0.50, 0.80]
    assert protocol["axes"] == ["x", "y", "z"]
    assert len(protocol["levels_g"]) * len(protocol["axes"]) == 15
    assert parser().parse_args([]).stability_test is False
    assert parser().parse_args(["--stability-test"]).stability_test is True


def test_beam_weights_sum_to_one() -> None:
    assert abs(sum(CFG["planner"]["weights"].values()) - 1.0) < 1e-9


def test_heuristic_weights_are_the_fourteen_terms() -> None:
    """Los de `placing` NO suman 1, y no tienen que hacerlo.

    El score se normaliza dividiendo por la suma de pesos, así que lo que hay que anclar
    es que están los catorce y ninguno negativo. Una clave mal escrita aquí no da error:
    `weight_vector()` la rellena con 0.0 y el término se apaga en silencio.
    """
    weights = CFG["heuristic"]["weights"]
    assert set(weights) == set(METRIC_NAMES), set(weights) ^ set(METRIC_NAMES)
    assert all(value >= 0.0 for value in weights.values())


def test_static_obstacles_stamp_the_named_tables_of_the_level() -> None:
    """El oráculo estampa DOS mesas por su nombre, y la de recogida sólo si se monta.

    Hoy ninguna de las dos pisa la huella del palé (60 mm de aire la de recogida, 50 mm
    la auxiliar), así que sin mover nada se estampan cero celdas: eso es lo primero que
    se comprueba, porque es lo que deja de ser cierto si alguien retoca `scene.yaml`.
    Luego se mueven las dos a mano sobre la cubierta para ejercitar el gate, que es la
    única lógica nueva y la que no tendría red de otro modo.
    """
    grid = _empty_pallet_map()
    ny, nx = grid.cells.shape
    deck_z = float(CFG["pallet"]["deck_thickness"])

    def stamp(source: str, cfg: dict) -> np.ndarray:
        cells = np.zeros((ny, nx))
        scene = SimpleNamespace(cfg=cfg, deck_z=deck_z, level=SimpleNamespace(source=source))
        _stamp_static_obstacles(scene, cells, grid.origin, grid.cell_size, nx, ny)
        return cells

    for source in SOURCES:
        assert stamp(source, CFG).max() == 0.0, f"{source}: una mesa invade la huella del palé"

    # Las dos corridas sobre la cubierta, sin solaparse entre sí: x[-0.11, 0.61] la de
    # recogida y x[0.70, 1.20] la auxiliar, ambas centradas en y=0.
    invaded = dict(CFG)
    invaded["table"] = {**CFG["table"], "center": [0.25, 0.0]}
    invaded["auxiliary_table"] = {**CFG["auxiliary_table"], "center": [0.95, 0.0]}
    row = round((0.0 - grid.origin[1]) / grid.cell_size)
    pick_col = round((0.25 - grid.origin[0]) / grid.cell_size)
    auxiliary_col = round((0.95 - grid.origin[0]) / grid.cell_size)
    expected = float(CFG["auxiliary_table"]["height"]) - deck_z

    on_belt = stamp("conveyor", invaded)
    assert on_belt[row, auxiliary_col] == expected, "la auxiliar existe en todos los niveles"
    assert on_belt[row, pick_col] == 0.0, "la mesa de recogida no se monta fuera de los 1x"

    on_table = stamp("table", invaded)
    assert on_table[row, auxiliary_col] == expected
    assert on_table[row, pick_col] == float(CFG["table"]["height"]) - deck_z


def _empty_pallet_map() -> Heightmap:
    """Un palé vacío con la rejilla y el origen de verdad de esta celda."""
    cell = float(CFG["heuristic"]["cell_size"])
    length, width = (float(v) for v in CFG["pallet"]["dims"])
    center_x, center_y = (float(v) for v in CFG["pallet"]["center"])
    nx, ny = round(length / cell), round(width / cell)
    return Heightmap(
        cells=np.zeros((ny, nx)),
        origin=(center_x - length / 2, center_y - width / 2),
        cell_size=cell,
    )


def test_the_adapter_translates_degrees_and_the_deck() -> None:
    """Las dos traducciones que fallan en SILENCIO: yaw y el convenio de alturas.

    `placing` habla grados y Z del mundo; este repo, radianes y metros sobre la cubierta.
    Ninguna de las dos peta si se olvida: las cajas acaban giradas 57° o flotando (o
    hundidas) exactamente la cota de la cubierta.
    """
    planner = ScorePlanner(CFG)
    deck_z = float(CFG["pallet"]["deck_thickness"])

    # Una caja mucho más larga que ancha sólo cabe girada en un palé estrecho: así se
    # fuerza un yaw de 90° sin tener que adivinar cuál elige la heurística.
    narrow = Heightmap(cells=np.zeros((60, 40)), origin=(0.0, -0.20), cell_size=0.01)
    plan = ScorePlanner(CFG).choose(
        PackageSpec("b", "std_m", (0.55, 0.20, 0.10), 2.0, np.zeros(3)), narrow
    )
    assert plan is not None
    assert abs(plan.yaw - np.pi / 2) < 1e-9, f"yaw={plan.yaw} no son 90° en radianes"

    # Altura de mapa 0 = la cubierta del palé, no el suelo.
    plan = planner.choose(
        PackageSpec("b", "std_m", (0.42, 0.30, 0.18), 4.2, np.zeros(3)),
        _empty_pallet_map(),
    )
    assert plan is not None
    z_base = float(plan.position[2]) - 0.18 / 2
    assert abs(z_base - deck_z) < 1e-6, f"z_base={z_base} debería ser deck_z={deck_z}"


def test_the_chosen_pose_lands_on_the_pallet() -> None:
    """Ida y vuelta de tipos: si el origen o la celda se traducen mal, se sale y se ve."""
    heightmap = _empty_pallet_map()
    dims = (0.42, 0.30, 0.18)
    plan = ScorePlanner(CFG).choose(
        PackageSpec("b", "std_m", dims, 4.2, np.zeros(3)), heightmap
    )
    assert plan is not None
    ny, nx = heightmap.cells.shape
    x0, y0 = heightmap.origin
    x1 = x0 + nx * heightmap.cell_size
    y1 = y0 + ny * heightmap.cell_size
    half_x, half_y = (dims[1] / 2, dims[0] / 2) if abs(plan.yaw) > 0.1 else (dims[0] / 2, dims[1] / 2)
    assert x0 - 1e-9 <= plan.position[0] - half_x and plan.position[0] + half_x <= x1 + 1e-9
    assert y0 - 1e-9 <= plan.position[1] - half_y and plan.position[1] + half_y <= y1 + 1e-9


def test_score_planner_returns_none_and_says_why() -> None:
    """Que no quepa es un resultado, no una excepción — y el porqué se registra."""
    planner = ScorePlanner(CFG)
    plan = planner.choose(
        PackageSpec("gigante", "std_m", (2.0, 2.0, 0.2), 4.2, np.zeros(3)),
        _empty_pallet_map(),
    )
    assert plan is None
    # Ni un candidato generado: no cabe ni en un palé vacío. `{}` NO es lo mismo que
    # "todos rechazados", y la diferencia es la que dice qué hacer con la caja.
    assert planner.last_reject == {}

    plan = planner.choose(
        PackageSpec("b", "std_m", (0.42, 0.30, 0.18), 4.2, np.zeros(3)),
        _empty_pallet_map(),
    )
    assert plan is not None
    assert 0.0 <= plan.score <= 1.0
    assert set(plan.breakdown) == set(METRIC_NAMES)
    assert planner.last_reject is None


def test_record_reads_the_pallet_frame_and_groups_the_layer() -> None:
    """`measure.Placement.position` va en el frame del PALÉ, no en el del mundo.

    Las dos formas de equivocarse aquí no dan error, sólo números plausibles:

      - Restarle `deck_z` a una Z que ya está sobre la cubierta deja los niveles en
        −0.144, y entonces una caja puesta en la cubierta sale como capa 2. `measure` le
        busca apoyo en la capa 1 en vez de en el palé y la traza dice 0 % de apoyo sobre
        un montón perfectamente plano.
      - Agrupar los niveles con `set()` en vez de con tolerancia cuenta cada caja de la
        misma capa como un nivel propio, y la capa sube de una en una.
    """
    planner = ScorePlanner(CFG)
    # Dos cajas asentadas en la cubierta, a alturas que difieren en micras.
    for dz in (0.0, 4e-6):
        planner.record(_placement(0.1, 0.1))
        planner._levels[-1] = 0.18 / 2 - 0.18 / 2 + dz    # base 0 en el frame del palé
    assert max(abs(level) for level in planner._levels) < 1e-5, planner._levels

    deck_z = float(CFG["pallet"]["deck_thickness"])
    assert planner._layer(deck_z) == 1                    # en la cubierta
    assert planner._layer(deck_z + 0.18) == 2             # encima de esas dos: capa 2
    planner._levels.append(0.18)
    assert planner._layer(deck_z + 0.36) == 3


def test_the_m_key_draws_the_height_map_grid() -> None:
    """La rejilla del visor: se enciende, cabe en el presupuesto y marca lo no observado.

    Sin visor no hace nada —por eso se puede llamar siempre—; con visor pinta un prisma
    por celda submuestreada, y las celdas que ninguna cámara vio salen con otro color,
    que es la única forma de verlas: guardan altura 0 igual que la cubierta libre.
    """
    geoms = [SimpleNamespace(rgba=np.zeros(4), pos=np.zeros(3)) for _ in range(2000)]
    viewer = SimpleNamespace(
        user_scn=SimpleNamespace(ngeom=0, maxgeom=2000, geoms=geoms),
        sync=lambda: None,
    )

    def init_geom(geom, *, type, size, pos, mat, rgba):
        geom.rgba = np.asarray(rgba, dtype=float)
        geom.pos = np.asarray(pos, dtype=float)

    fake_mujoco = SimpleNamespace(
        mjv_initGeom=init_geom,
        mjtGeom=SimpleNamespace(mjGEOM_BOX=6),
    )
    heightmap = _empty_pallet_map()
    observed = np.ones(heightmap.cells.shape, dtype=bool)
    observed[:20, :20] = False
    heightmap = Heightmap(heightmap.cells, heightmap.origin, heightmap.cell_size, observed)
    scene = SimpleNamespace(viewer=viewer, mujoco=fake_mujoco, cfg=CFG,
                            deck_z=float(CFG["pallet"]["deck_thickness"]),
                            show_heightmap=False)

    draw_heightmap(scene, heightmap)
    assert viewer.user_scn.ngeom == 0, "apagado no pinta nada"

    scene.show_heightmap = True                      # esto es lo que hace la tecla `m`
    draw_heightmap(scene, heightmap)
    painted = viewer.user_scn.ngeom
    assert 0 < painted <= HEIGHTMAP_BUDGET, painted
    colours = {tuple(np.round(geoms[i].rgba, 3)) for i in range(painted)}
    assert len(colours) == 2, f"observado y no observado deberían distinguirse: {colours}"
    # Y todo cae sobre la huella del palé, no en el suelo de al lado.
    x0, y0 = heightmap.origin
    ny, nx = heightmap.cells.shape
    for i in range(painted):
        assert x0 <= geoms[i].pos[0] <= x0 + nx * heightmap.cell_size
        assert y0 <= geoms[i].pos[1] <= y0 + ny * heightmap.cell_size
        assert geoms[i].pos[2] >= scene.deck_z

    scene.viewer = None
    draw_heightmap(scene, heightmap)                 # sin visor: ni pincha ni corta


def test_the_m_key_does_not_leave_mujoco_com_spheres_on() -> None:
    """`M` YA era el atajo de MuJoCo para "Center of Mass", y pinta esferas blancas.

    Las veintiséis letras están cogidas por las banderas del visor, así que la rejilla no
    tiene ninguna libre: se queda con la tecla y deja la bandera de MuJoCo apagada. Si se
    comparte, una pulsación enciende las esferas y apaga la rejilla, que es justo el
    síntoma que hay que evitar.
    """
    import mujoco

    assert mujoco.mjVISSTRING[mujoco.mjtVisFlag.mjVIS_COM][2].upper() == "M", (
        "MuJoCo ha cambiado el atajo; revisa si la rejilla puede recuperar la tecla"
    )

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1       # como si el visor la hubiera puesto
    viewer = SimpleNamespace(
        opt=opt,
        user_scn=SimpleNamespace(ngeom=0, maxgeom=16, geoms=[]),
        sync=lambda: None,
    )
    scene = SimpleNamespace(viewer=viewer, mujoco=mujoco, cfg=CFG,
                            deck_z=0.144, show_heightmap=False)
    draw_heightmap(scene, _empty_pallet_map())
    assert opt.flags[mujoco.mjtVisFlag.mjVIS_COM] == 0


def test_the_table_stages_one_box_at_a_time() -> None:
    """La mesa prepara UN bulto, y el siguiente entra cuando se llevan el anterior.

    No es una simplificación: es lo único que cabe. La mesa mide 0.72 x 0.68 m útiles y
    dos `std_m` sólo entran con 80 mm de holgura como mucho; con esa holgura, al extraer
    uno el que va en la mano barre al vecino y lo tira al suelo (385 mm medidos), y la
    holgura que haría falta para no tocarlo ya no cabe en la mesa. Con la mesa llena, eso
    salía como que al brazo se le escurría la caja.

    Sólo se ve con movimiento real: en fast-forward el brazo teletransporta entre
    waypoints y no barre nada.
    """
    from src.cell.table import TableSupply

    boxes = [
        SimpleNamespace(index=i, dims_m=dims, package_id=f"b{i}")
        for i, dims in enumerate([(0.42, 0.30, 0.18), (0.24, 0.18, 0.10),
                                  (0.55, 0.36, 0.26)])
    ]
    puestas: dict[int, tuple[float, float, float]] = {}
    scene = SimpleNamespace(
        cfg=CFG, boxes=boxes,
        level=SimpleNamespace(yaw_jitter_deg=0.0, pos_jitter_m=0.0),
        rng=np.random.default_rng(0),
        place_box=lambda index, pos, yaw: puestas.__setitem__(index, pos),
        settle_until_rest=lambda: None,
    )
    supply = TableSupply(scene)
    supply.stage(scene)
    assert list(puestas) == [0], f"la mesa debe preparar uno solo: {list(puestas)}"

    cfg = CFG["table"]
    cx, cy = (float(v) for v in cfg["center"])
    half_x, half_y, _ = (float(v) for v in cfg["size"])
    for index, (x, y, _z) in puestas.items():
        length, width, _ = boxes[index].dims_m
        # Entero dentro de la mesa: uno a medias sobre el canto se cae, y lo que se mide
        # despues es una caja en el suelo.
        assert cx - half_x <= x - length / 2 and x + length / 2 <= cx + half_x, index
        assert cy - half_y <= y - width / 2 and y + width / 2 <= cy + half_y, index

    # El segundo entra al presentarlo, no antes, y en el sitio que dejó el primero.
    assert supply.present(scene) == "b0"
    assert list(puestas) == [0], "presentar el que ya está puesto no mueve nada"
    supply.release(scene)                      # el brazo se lo llevó
    assert supply.present(scene) == "b1"
    assert list(puestas) == [0, 1]
    assert puestas[1][:2] == puestas[0][:2]


def test_depth_fuses_into_the_height_map() -> None:
    """La fusión, sin MuJoCo: un fotograma sintético cae en la celda que le toca.

    Cámara cenital a 2 m mirando hacia abajo sobre un palé de 0.20x0.20. La mitad de la
    imagen ve una tapa a 0.30 m y la otra mitad la cubierta a 0.144. Lo que NO se ve
    tiene que quedar `observed=False`, que es la diferencia entre "no hay nada" y "no
    lo sé", y es justo lo que el oráculo no puede decir.
    """
    width = height = 128
    fovy, cam_z, surface_z = 45.0, 0.60, 0.30
    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    intrinsics = CameraIntrinsics(fx=fy, fy=fy, cx=(width - 1) / 2, cy=(height - 1) / 2,
                                  width=width, height=height, fovy_deg=fovy)
    # Cenital: la cámara mira por su −Z, que con esta rotación es el −Z del mundo.
    pose = CameraPose(pos=np.array([0.10, 0.10, cam_z]), rot=np.eye(3))
    depth = np.full((height, width), cam_z - surface_z, dtype=np.float32)
    # Una franja central que ninguna cámara ve. Va en el medio a propósito: cae DENTRO
    # de la huella del palé, que es donde la distinción importa. Y tiene que ser MÁS
    # ANCHA que una celda (≈1.9 mm por fila aquí, celda de 20 mm) o las filas de al lado
    # rellenan la celda igual y el hueco no llega a existir.
    depth[52:78, :] = np.nan

    points = unproject_depth(depth, intrinsics, pose, min_upward_nz=0.45)
    hmap = rasterize_points(points, origin_xy=(0.0, 0.0), length=0.20, width=0.20,
                            resolution=0.02, z_min=-0.005, z_max=2.5)
    fused = fuse_max([hmap, hmap])               # fusionar consigo mismo no cambia nada

    seen = fused.heights[fused.observed]
    assert seen.size, "no se observó ni una celda"
    assert np.allclose(seen, surface_z, atol=0.005), (float(seen.min()), float(seen.max()))
    assert not fused.observed.all(), "la franja NaN tendría que quedar sin observar"
    # Altura 0 y sin observar NO es lo mismo que cubierta libre: es lo que aporta medir.
    assert np.all(fused.heights[~fused.observed] == 0.0)


def test_oracle_is_any_stub_for_every_combination() -> None:
    for flags in itertools.product((False, True), repeat=4):
        assert oracle_for(*flags) is any(flags)


def test_an_aborted_episode_is_unsuccessful_without_inventing_a_failure() -> None:
    scene = SimpleNamespace(
        level=SimpleNamespace(id=11, source="table"),
        boxes=[object()], clock=1.25, oracle=False,
    )
    result = _aborted(7, scene)
    assert result.success is False
    assert result.failure is None
    assert result.metrics["aborted"] is True
    assert result.task == "table"


def test_sigterm_unwinds_like_ctrl_c() -> None:
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    try:
        install_termination_unwind()
        assert signal.getsignal(signal.SIGTERM) is _request_unwind
        try:
            _request_unwind(signal.SIGTERM, None)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("SIGTERM tiene que deshacer como Ctrl-C")
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigterm_runs_the_finally_that_closes_an_episode() -> None:
    """El default de SIGTERM mata sin finally; el CLI no puede permitírselo."""
    import signal
    import subprocess
    import tempfile
    import textwrap

    marker = Path(tempfile.mkdtemp()) / "cleaned"
    child = textwrap.dedent(f"""\
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(REPO)!r})
        from scripts.palletize import install_termination_unwind
        install_termination_unwind()
        try:
            print("ready", flush=True)
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            Path(sys.argv[1]).write_text("ok")
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        if line != "ready":
            err = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"el hijo no arrancó: {line!r}\n{err}")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=8) != -signal.SIGTERM
        assert marker.read_text() == "ok"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_planners_keep_their_contract() -> None:
    heightmap = Heightmap(np.zeros((20, 30)), (0.0, 0.0), 0.04)
    spec = PackageSpec("box-1", "std_m", (0.42, 0.30, 0.18), 4.2, np.zeros(3))
    naive = GridPlanner(CFG).choose(spec, heightmap)
    scored = BeamPlanner(CFG).choose(spec, heightmap)
    assert naive is not None and naive.score == 0.0 and naive.breakdown == {}
    assert scored is not None and 0.0 <= scored.score <= 1.0 and scored.breakdown
    impossible = PackageSpec("huge", "huge", (2.0, 2.0, 0.2), 1.0, np.zeros(3))
    assert GridPlanner(CFG).choose(impossible, heightmap) is None
    assert BeamPlanner(CFG).choose(impossible, heightmap) is None


def test_cog_counts_boxes_outside_tolerance() -> None:
    good = _placement(-0.05, placed=True)
    bad = _placement(0.05, placed=False, cog=(0.02, 0.0, 0.0))
    only_good = measure.pallet_state([good])
    both = measure.pallet_state([good, bad])
    assert both.mass_kg > only_good.mass_kg
    assert both.cog[0] > only_good.cog[0]


def test_a_box_left_on_the_table_is_not_on_the_pallet() -> None:
    scene = SimpleNamespace(pallet_dims=(1.2, 0.8))
    on = _placement(0.0, 0.0, placed=False)
    off = _placement(0.0, 0.7, placed=False)
    assert measure.on_pallet(scene, [on, off]) == [on]


def test_stability_margin_uses_the_support_polygon() -> None:
    base = [(-0.05, 0.0, 0.10, 0.06), (0.05, 0.0, 0.10, 0.06)]
    assert measure.support_polygon(base) == (-0.10, 0.10, -0.03, 0.03)
    centered = measure.stability_margin(0.0, 0.0, 0.05, base)
    assert round(centered, 9) == round(0.03 - 0.28 * 0.05, 9)
    assert measure.stability_margin(0.0, 0.0, 0.10, base) < centered
    assert measure.stability_margin(0.0, 0.05, 0.05, base) < 0.0
    assert measure.stability_margin(0.0, 0.0, 0.0, []) == 0.0


def test_run_config_names_source_level_and_real_pallet() -> None:
    scene = SimpleNamespace(
        cfg=CFG,
        level=Level(11, "table", "mesa", 1, ("std_m",)),
        boxes=[SimpleNamespace(
            package_id="box-1", type_name="std_m", dims_m=(0.42, 0.30, 0.18),
            mass_kg=4.2,
        )],
    )
    config = run_config(scene)
    assert config["pallet_size_m"] == [1.2, 0.8]
    assert config["source"] == "table" and config["level_name"] == "mesa"
    assert config["auxiliary_table"] == CFG["auxiliary_table"]


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    measure.demo()
    print(f"{len(tests)} comprobaciones pasadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
