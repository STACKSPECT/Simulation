"""Pruebas físicas de la celda. Arrancan MuJoCo, pero no usan red."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.cell import tcp_frame  # noqa: E402
from src.cell.arm import ArmController  # noqa: E402
from src.cell.conveyor import Belt, make_supply  # noqa: E402
from src.cell.render import VIEWS  # noqa: E402
from src.cell.scene import build_scene, load_configs  # noqa: E402
from src.cell.truck import TruckSupply  # noqa: E402
from src.contracts import Observation, PackageSpec, PlacementPlan  # noqa: E402
from src.episode import _pick, _place  # noqa: E402
from src.vision.gauge import WristGauge  # noqa: E402
from src.vision.oracle import OracleDetector  # noqa: E402


def test_scenes_compile_for_all_sources() -> None:
    for level in (11, 21, 31):
        scene = build_scene(level_id=level, seed=3, simplified=True)
        try:
            supply = make_supply(scene)
            supply.stage(scene)
            assert scene.model.nbody > 0
            assert scene.level.id == level
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
            failed = [
                target for target in targets
                if arm.solve_ik(target[:3], tcp_frame(target[:3], target[3]).rotation) is None
            ]
            assert not failed, f"nivel {level}: poses fuera de alcance: {failed}"
        finally:
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
        assert _pick(arm, scene, box, observation) is None
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


def _one_box_scene(type_name: str):
    """Mesa, un solo paquete, sin jitter: el caso aislado del issue #13."""
    cfg = copy.deepcopy(load_configs())
    for row in cfg["levels"]:
        if int(row["id"]) == 11:
            row["n_packages"] = 1
            row["types"] = [type_name]
            row["yaw_jitter_deg"] = 0
            row["pos_jitter_m"] = 0
            row["cog"] = "centred"
    scene = build_scene(cfg, level_id=11, seed=1, simplified=True)
    supply = make_supply(scene)
    supply.stage(scene)
    scene.supply = supply
    return scene, supply


def _observation(scene, box, yaw: float) -> Observation:
    position, _ = scene.box_pose(box.index)
    return Observation(
        package_id=box.package_id,
        position=position,
        yaw=yaw,
        dims_guess=box.dims_m,
        confidence=1.0,
    )


def _plan_on_pallet(scene, spec: PackageSpec, yaw: float) -> PlacementPlan:
    px, py = scene.pallet_center
    return PlacementPlan(
        position=np.array([px, py, scene.deck_z + spec.dims_m[2] / 2]),
        yaw=yaw,
        layer=1,
        slot="test",
        score=1.0,
        predicted_support=1.0,
        breakdown={},
    )


def _assert_cups_on_carton(scene, arm, box) -> None:
    grasp = arm.grasp
    assert grasp is not None and grasp.cups
    centre = scene.data.xpos[scene.body_id(box.index)][:2]
    half = np.asarray(box.dims_m[:2], dtype=float) / 2
    leftover = 0.003
    yaw = scene.box_yaw(box.index)
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    cfg = scene.cfg["vacuum"]
    active = {(round(x, 4), round(y, 4)) for x, y in grasp.cups}
    radius = float(cfg["cup_radius"])
    for column in range(cfg["columns"]):
        x = (column - (cfg["columns"] - 1) / 2) * cfg["pitch_x"]
        for row in range(cfg["rows"]):
            y = (row - (cfg["rows"] - 1) / 2) * cfg["pitch_y"]
            if (round(x, 4), round(y, 4)) not in active:
                continue
            geom = scene.mujoco.mj_name2id(
                scene.model, scene.mujoco.mjtObj.mjOBJ_GEOM, f"cup_{column}_{row}"
            )
            assert geom >= 0
            delta = scene.data.geom_xpos[geom][:2] - centre
            local = np.array([
                cosine * delta[0] - sine * delta[1],
                sine * delta[0] + cosine * delta[1],
            ])
            assert abs(local[0]) + radius <= half[0] + leftover, (box.type_name, column, row, local)
            assert abs(local[1]) + radius <= half[1] + leftover, (box.type_name, column, row, local)


def test_engaged_cups_land_on_small_and_large_cartons() -> None:
    """El offset del bloque tiene que poner las ventosas SOBRE el cartón, no al lado."""
    for type_name, yaw in (("book_s", 0.0), ("book_s", math.pi / 2), ("std_m", 0.0)):
        scene, supply = _one_box_scene(type_name)
        arm = ArmController(scene)
        arm.fast_forward = True
        try:
            box = scene.boxes[0]
            position, _ = scene.box_pose(box.index)
            scene.place_box(box.index, position, yaw)
            scene.settle(0.15)
            supply.current = box.index
            observation = _observation(scene, box, yaw)
            assert _pick(arm, scene, box, observation) is None
            if type_name == "book_s":
                assert arm.grasp is not None and arm.grasp.tool_offset != (0.0, 0.0)
            else:
                assert arm.grasp is not None and arm.grasp.tool_offset == (0.0, 0.0)
            _assert_cups_on_carton(scene, arm, box)
        finally:
            if arm.held is not None:
                arm.release()
            scene.close()


def test_pick_and_place_keep_cup_block_offset_consistent() -> None:
    """El error de un book_s era ~45 mm; no se tapa subiendo la tolerancia de colocación."""
    cases = (
        ("book_s", 0.0, 0.0),
        ("book_s", math.pi / 4, math.pi / 4),
        ("book_s", math.pi / 2, 0.0),
        ("std_m", 0.0, 0.0),
        ("std_m", math.pi / 2, math.pi / 2),
    )
    limit = 0.008
    for type_name, pick_yaw, place_yaw in cases:
        scene, supply = _one_box_scene(type_name)
        arm = ArmController(scene)
        arm.fast_forward = True
        try:
            box = scene.boxes[0]
            position, _ = scene.box_pose(box.index)
            scene.place_box(box.index, position, pick_yaw)
            scene.settle(0.15)
            supply.current = box.index
            observation = _observation(scene, box, pick_yaw)
            assert _pick(arm, scene, box, observation) is None, (type_name, pick_yaw)
            spec = PackageSpec(
                box.package_id, box.type_name, box.dims_m, box.mass_kg,
                np.asarray(box.cog_offset_m, dtype=float),
            )
            plan = _plan_on_pallet(scene, spec, place_yaw)
            failure, _drift = _place(arm, scene, box, spec, plan, [box])
            assert failure is None, (type_name, pick_yaw, place_yaw, failure)
            actual, _ = scene.box_pose(box.index)
            error_xy = float(np.linalg.norm(actual[:2] - plan.position[:2]))
            assert error_xy < limit, (
                f"{type_name} pick={pick_yaw:.2f} place={place_yaw:.2f}: "
                f"{error_xy * 1000:.1f} mm"
            )
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
