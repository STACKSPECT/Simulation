"""Pruebas físicas de la celda. Arrancan MuJoCo, pero no usan red."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.cell import tcp_frame  # noqa: E402
from src.cell.arm import ArmController  # noqa: E402
from src.cell.conveyor import Belt, make_supply  # noqa: E402
from src.cell.render import VIEWS  # noqa: E402
from src.cell.scene import build_scene  # noqa: E402
from src.cell.truck import TruckSupply  # noqa: E402
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
