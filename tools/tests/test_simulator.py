import math

import mujoco
import numpy as np
import pytest

from stable_pallet.cli import build_parser
from stable_pallet.scenario import load_scenario
from stable_pallet.simulator import (
    HeldPackage,
    PalletizingSimulator,
    _hide_viewer_mass_overlays,
    build_mjcf,
)
from stable_pallet.suction import SuctionArray


def test_mujoco_cell_compiles_with_all_packages_and_suction_constraints() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    model = mujoco.MjModel.from_xml_string(build_mjcf(scenario, simplified_graphics=True))
    assert model.neq == len(scenario.packages) + 1
    assert model.nu == 6
    assert model.nbody >= len(scenario.packages) + 5
    assert model.body("pallet").id >= 0
    assert model.joint("pallet_x").id >= 0
    assert model.joint("pallet_z").id >= 0
    assert model.joint("pallet_rx").id >= 0
    assert model.joint("pallet_ry").id >= 0
    assert model.body("beam").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "pallet_anchor") >= 0


def test_demo_packages_fit_ur10e_payload_with_tool() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    assert PalletizingSimulator.tool_mass_kg == SuctionArray().mass
    assert max(package.mass for package in scenario.packages) + PalletizingSimulator.tool_mass_kg <= (
        PalletizingSimulator.ur10e_payload_kg
    )
    assert scenario.pallet.width == 1.2
    assert scenario.pallet.depth == 0.8


def test_detailed_cell_is_default_and_simplified_mode_is_selectable() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    detailed = build_mjcf(scenario)
    simplified = build_mjcf(scenario, simplified_graphics=True)

    assert '<mesh name="upperarm_3"' in detailed
    assert 'name="vgp20_body"' in detailed
    assert 'name="pallet_top_0"' in detailed
    assert 'name="package_label_0"' in detailed
    assert '<mesh name="upperarm_3"' not in simplified
    assert 'name="pallet_top_0"' not in simplified

    args = build_parser().parse_args(["simulate", "--simplified-graphics"])
    assert args.simplified_graphics is True


def test_viewer_hides_mass_debug_overlays() -> None:
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

    _hide_viewer_mass_overlays(mujoco, option)

    assert not option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]
    assert not option.flags[mujoco.mjtVisFlag.mjVIS_COM]


def test_ur10e_ik_reaches_infeed_and_pallet() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True
    )
    try:
        assert simulator._solve_ik((-0.20, -0.66, 0.80), 0).shape == (6,)
        assert simulator._solve_ik((0.60, 0.00, 0.60), 90).shape == (6,)
    finally:
        simulator.close()


def test_the_cell_says_what_it_is_doing_while_nothing_moves() -> None:
    """The planner stops the cell for seconds; the panel reads this to prove it is alive."""
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True
    )
    try:
        simulator.set_viewer_status("caja 3", "error 2.1 mm")

        with simulator.busy("Planificando…"):
            assert simulator.controls.activity == "Planificando…"
            assert simulator.viewer_status[0] == "Planificando…"

        assert simulator.controls.activity == ""
        assert simulator.viewer_status == ("caja 3", "error 2.1 mm")
    finally:
        simulator.close()


def test_a_carried_package_keeps_its_cardboard_look() -> None:
    """Re-parenting a package onto the tool must not strip its decoration."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    held = build_mjcf(scenario, HeldPackage(0, (0.0, 0.0, -0.2), (1.0, 0.0, 0.0, 0.0)))

    carried = held.split('<body name="tool"')[1]
    assert 'name="package_0"' in carried
    assert 'name="package_tape_0"' in carried
    assert 'name="package_label_0"' in carried


def test_rebuilding_the_cell_keeps_the_colours_it_had_set() -> None:
    """Recompiling resets every geom to the XML, which would undo the pick's own staging."""
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True
    )
    try:
        simulator._set_package_pose(0, (-0.20, -0.66, 0.80))
        _column, _row, cup_x, cup_y = SuctionArray().cup_offsets()[0]
        simulator._show_active_cups(((cup_x, cup_y),))
        before = {
            name: simulator.model.geom_rgba[simulator._geom_id(name)].copy()
            for name in simulator._appearance_geom_names()
        }

        simulator.rebuild(None)

        for name, rgba in before.items():
            assert (simulator.model.geom_rgba[simulator._geom_id(name)] == rgba).all(), name
        # The staging is what makes the package visible at all, so this is not cosmetic.
        assert before["package_geom_0"][3] == 1.0
    finally:
        simulator.close()


def test_the_viewer_can_be_asked_for_from_the_command_line() -> None:
    args = build_parser().parse_args(["simulate", "--viewer"])
    assert args.viewer is True
    assert args.measure_com is True
    assert args.precise_com is False
    assert build_parser().parse_args(["simulate", "--no-measure-com"]).measure_com is False
    assert build_parser().parse_args(["simulate", "--precise-com"]).precise_com is True


def test_the_shake_command_is_wired_on_the_cli() -> None:
    args = build_parser().parse_args(["shake", "--instant-place", "--no-measure-com"])
    assert args.command == "shake"
    assert args.instant_place is True
    assert args.measure_com is False
    assert args.output == "artifacts/shake.json"


def test_grasp_offset_reaches_the_world_through_the_tool_frame() -> None:
    """The tool hangs upside down, so its y axis opposes the world's.

    Cancelling the cup-block offset straight in world coordinates doubles it on that
    axis, which silently drops the outer row of cups off the edge of the carton while
    every purely geometric check in `suction.py` still passes.
    """
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True
    )
    try:
        offset = (0.03, 0.02)
        # `reference_rotation` is read off the compiled model at the home pose, so it is
        # diag(1, -1, -1) only to a few microns.
        shift = simulator._grasp_offset_world(offset, 0)
        assert shift == pytest.approx((0.03, -0.02), abs=1e-5)
        turned = simulator._grasp_offset_world(offset, 90)
        assert turned == pytest.approx((0.02, 0.03), abs=1e-5)
    finally:
        simulator.close()


def test_engaged_cups_land_on_the_carton_once_the_arm_cancels_the_offset() -> None:
    """End to end through the real tool rotation, for every package and both yaws."""
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True
    )
    try:
        array = SuctionArray()
        for package in simulator.scenario.packages:
            # Pickup is always at yaw 0; 90° turns the tool and the carton together.
            grip = array.plan(package, 0)
            for yaw in (0, 90):
                width, depth, _ = package.oriented_size(yaw)
                shift = np.asarray(simulator._grasp_offset_world(grip.tool_offset, yaw))
                rotation = simulator._rotation_z(math.radians(yaw)) @ simulator.reference_rotation
                for cup in grip.active_cups:
                    # Where the cup ends up relative to the carton centre, once the arm has
                    # moved by -shift to put the block over it.
                    landed = rotation[:2, :2] @ np.asarray(cup) - shift
                    assert abs(landed[0]) + array.cup_radius <= width / 2 + 1e-9, package.id
                    assert abs(landed[1]) + array.cup_radius <= depth / 2 + 1e-9, package.id
    finally:
        simulator.close()


def test_the_arm_lands_the_cup_block_on_the_carton() -> None:
    """IK has to actually put those cups on the cardboard, not just the offset math.

    AMZ-BOOK-S is smaller than the 264 x 184 mm housing: without the offset the outer
    cups hang off one edge, which is the pick that looks like a single-sided grab.
    """
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
    )
    try:
        array = SuctionArray()
        index = 0  # AMZ-BOOK-S
        package = simulator.scenario.packages[index]
        grip = array.plan(package, 0)
        assert grip.tool_offset != (0.0, 0.0)
        infeed_x, infeed_y = simulator.scenario.simulation.infeed_position
        feed_z = simulator.scenario.simulation.infeed_height + package.size[2] / 2 + 0.001
        simulator._set_package_pose(index, (infeed_x, infeed_y, feed_z))
        simulator._step(0.2)
        offset_x, offset_y = simulator._grasp_offset_world(grip.tool_offset, 0)
        pick = simulator._package_top_center(index)
        simulator._move_tool(
            float(pick[0]) - offset_x,
            float(pick[1]) - offset_y,
            float(pick[2] + simulator.cup_gap),
            0,
            0.60,
        )

        body = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
        centre = simulator.data.xpos[body][:2]
        half_w, half_d = package.size[0] / 2, package.size[1] / 2
        # `_move_tool` accepts 2.5 mm of cartesian leftover; a cup on the lip still seals.
        leftover = 0.003
        active = {(round(x, 4), round(y, 4)) for x, y in grip.active_cups}
        for column, row, x, y in array.cup_offsets():
            if (round(x, 4), round(y, 4)) not in active:
                continue
            cup = simulator.data.geom_xpos[simulator._geom_id(f"cup_{column}_{row}")][:2]
            delta = np.abs(cup - centre)
            assert delta[0] + array.cup_radius <= half_w + leftover, (package.id, column, row, delta)
            assert delta[1] + array.cup_radius <= half_d + leftover, (package.id, column, row, delta)
    finally:
        simulator.close()


def test_fast_forward_lands_on_the_same_pose_the_trajectory_would_reach() -> None:
    """Skipping the transit must not cost accuracy: the cell still settles on arrival."""
    target = (-0.20, -0.66, 0.95)
    poses = {}
    for fast_forward in (False, True):
        simulator = PalletizingSimulator(
            load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
        )
        try:
            simulator.controls.fast_forward = fast_forward
            start = simulator.data.time
            simulator._move_tool(*target, 0)
            poses[fast_forward] = (
                np.asarray(simulator.data.site_xpos[simulator.site_id]).copy(),
                simulator.data.time - start,
            )
        finally:
            simulator.close()

    for reached, _elapsed in poses.values():
        assert np.linalg.norm(reached - np.asarray(target)) < 0.0025
    # The saving is the travel: the same waypoint, reached in a fraction of the time.
    assert poses[True][1] < poses[False][1] / 4


def test_fast_forward_carries_a_sealed_carton_with_the_arm() -> None:
    """A welded package is a free body: jumping the arm alone would tear it across the cell."""
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
    )
    try:
        index = 2
        package = simulator.scenario.packages[index]
        infeed_x, infeed_y = simulator.scenario.simulation.infeed_position
        feed_z = simulator.scenario.simulation.infeed_height + package.size[2] / 2 + 0.001
        simulator._set_package_pose(index, (infeed_x, infeed_y, feed_z))
        simulator._step(0.2)
        pick = simulator._package_top_center(index)
        simulator._move_tool(float(pick[0]), float(pick[1]), float(pick[2] + simulator.cup_gap), 0)
        simulator._set_suction(index, True)

        body = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
        tool = simulator.data.xpos[simulator.tool_body_id]
        before = simulator.data.xpos[body] - tool

        simulator.controls.fast_forward = True
        simulator._move_tool(0.45, -0.10, 1.05, 90)

        after = simulator.data.xpos[body] - simulator.data.xpos[simulator.tool_body_id]
        assert np.linalg.norm(after) == pytest.approx(np.linalg.norm(before), abs=0.004)
    finally:
        simulator.close()


def test_speed_paces_the_viewer_and_fast_forward_does_not_touch_it() -> None:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
    )
    try:
        timestep = simulator.model.opt.timestep
        assert simulator.frame_pause == pytest.approx(timestep)
        simulator.controls.speed = 4.0
        assert simulator.frame_pause == pytest.approx(timestep / 4)
        simulator.controls.fast_forward = True
        assert simulator.frame_pause == pytest.approx(timestep / 4)
        simulator.controls.speed = 0.0
        assert simulator.frame_pause == 0.0

        simulator.controls.speed = 1.0
        with simulator.viewer_fast_forward():
            assert simulator.frame_pause == 0.0
        assert simulator.frame_pause == pytest.approx(timestep)
    finally:
        simulator.close()


def test_the_viewer_flags_are_wired_on_the_command_line() -> None:
    args = build_parser().parse_args(
        ["simulate", "--speed", "0.5", "--fast-forward", "--show-com", "--show-estimated-com"]
    )
    assert args.speed == 0.5
    assert args.fast_forward is True
    assert args.show_com is True
    assert args.show_estimated_com is True


def test_the_viewer_weld_is_stiff() -> None:
    """`--no-measure-com` holds with a weld; slack rotation is what lets a knock flip a carton."""
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True)
    )
    equality = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_0")
    assert model.eq_data[equality, 10] == pytest.approx(1.0)


def test_a_held_carton_does_not_spin_on_the_cups_when_knocked() -> None:
    """Sixteen cups on a 264 x 184 mm footprint hold a carton flat.

    The weld carries the package whenever the centre of mass is not being measured, which
    is every viewer run. At a slack torquescale a 5 Nm knock flips the carton right over
    on the tool, and it then lands badly enough to strand the planner on the next box.
    """
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=False
    )
    try:
        mujoco_module = simulator.mujoco
        index = 4  # AMZ-HEAVY-L, the heaviest carton in the mix
        package = simulator.known_packages[index]
        body = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")

        simulator._set_package_pose(index, (-0.20, -0.66, 0.80))
        simulator._step(0.2)
        pick = simulator._package_top_center(index)
        simulator._move_tool(float(pick[0]), float(pick[1]), float(pick[2] + simulator.cup_gap), 0)
        simulator._set_suction(index, True)
        # The knock only means anything once the carton is off the conveyor and the cups are
        # the only thing holding it; resting on the rollers it cannot turn over at all.
        simulator._move_tool(float(pick[0]), float(pick[1]), 1.05, 0, 0.5)

        worst = 0.0
        for step in range(400):
            simulator.data.xfrc_applied[body][3:6] = [5.0, 0.0, 0.0] if step < 200 else [0.0, 0.0, 0.0]
            mujoco_module.mj_step(simulator.model, simulator.data)
            upright = simulator.data.xmat[body].reshape(3, 3)[2, 2]
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, upright)))))

        assert worst < 5.0, f"{package.id} span {worst:.1f} deg on the cups"
    finally:
        simulator.close()


def _truck_scenario(seed: int = 0, count: int = 6):
    from stable_pallet.trial import build_generated_scenario

    return build_generated_scenario(count, seed)


def test_a_configured_trailer_replaces_the_infeed_conveyor() -> None:
    """The cartons come from one place or the other; the cell never shows both."""
    conveyor = build_mjcf(load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True)
    trailer = build_mjcf(_truck_scenario(), simplified_graphics=True)

    assert 'name="infeed"' in conveyor
    assert "truck_deck" not in conveyor
    assert 'name="truck_deck"' in trailer
    assert 'name="infeed"' not in trailer
    assert 'name="dock"' in trailer  # the camera that looks in through the open doors


def test_the_cell_always_takes_the_carton_with_nothing_on_top_of_it() -> None:
    """The load is stacked, so the only legal pick is the highest top left in the bay."""
    from stable_pallet.truck import unload_order

    simulator = PalletizingSimulator(_truck_scenario(), simplified_graphics=True, measure_com=False)
    try:
        simulator.stage_truck_load()
        remaining = list(range(len(simulator.scenario.packages)))
        taken = []
        while remaining:
            index = simulator.next_pick(remaining)
            tops = {other: simulator._package_top_center(other)[2] for other in remaining}
            assert tops[index] == max(tops.values())
            remaining.remove(index)
            taken.append(simulator.scenario.packages[index].id)
        assert taken == [slot.package_id for slot in unload_order(simulator.truck_slots)]
    finally:
        simulator.close()


def test_the_arm_reaches_every_carton_of_a_random_trailer_load() -> None:
    """The bay is only useful if the whole of it is inside the working envelope."""
    from stable_pallet.generator import generate_boxes
    from stable_pallet.truck import TruckBay, plan_truck_load

    simulator = PalletizingSimulator(_truck_scenario(), simplified_graphics=True, measure_com=False)
    array = SuctionArray()
    bay = TruckBay()
    try:
        for seed in range(6):
            packages = generate_boxes(6, seed=seed)
            for slot in plan_truck_load(packages, bay, seed=seed):
                grip = array.plan(packages[slot.index], 0)
                offset_x, offset_y = simulator._grasp_offset_world(grip.tool_offset, slot.yaw)
                pick_z = slot.top + simulator.cup_gap
                for height in (pick_z, max(0.96, pick_z + 0.18)):
                    pose = (slot.center[0] - offset_x, slot.center[1] - offset_y, height)
                    assert simulator._solve_ik(pose, slot.yaw).shape == (6,), (seed, slot.package_id)
    finally:
        simulator.close()


def test_the_trailer_body_stops_a_carton_instead_of_letting_it_through() -> None:
    """The truck is structure: a carton shoved at the side of the body hits it."""
    simulator = PalletizingSimulator(_truck_scenario(), simplified_graphics=True, measure_com=False)
    try:
        bay = simulator.truck
        assert bay is not None
        index = 0
        package = simulator.scenario.packages[index]
        wall = simulator._geom_id("truck_far_wall")
        inner_face = float(
            simulator.model.geom_pos[wall][1] + simulator.model.geom_size[wall][1]
        )
        start_y = bay.origin[1] + package.size[1]
        simulator._set_package_pose(
            index,
            (bay.center[0], start_y, bay.floor_height + package.size[2] / 2 + 0.001),
        )
        body = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")

        for step in range(900):
            simulator.data.xfrc_applied[body][:3] = (0.0, -60.0, 0.0) if step < 400 else (0.0, 0.0, 0.0)
            mujoco.mj_step(simulator.model, simulator.data)
        simulator.data.xfrc_applied[body][:] = 0.0

        centre_y = float(simulator.data.xpos[body][1])
        assert centre_y < start_y - 0.05, "the shove did not move the carton at all"
        assert centre_y - package.size[1] / 2 > inner_face - 0.01, "the carton went through the body"
        assert centre_y - package.size[1] / 2 < inner_face + 0.06, "the carton never reached the body"
    finally:
        simulator.close()
