import mujoco
import numpy as np
import pytest

from stable_pallet.com_probe import MEASUREMENT_OFFSETS, ComProbe, ProbeConfig
from stable_pallet.scenario import load_scenario
from stable_pallet.simulator import HeldPackage, PalletizingSimulator, build_mjcf

HEAVY_INDEX = 4  # AMZ-HEAVY-L, 8.5 kg with an off-centre load


def _reset_arm(simulator: PalletizingSimulator) -> None:
    simulator.data.qpos[simulator.arm_qpos] = simulator.home
    simulator.data.ctrl[:] = simulator.home
    simulator.data.qvel[:] = 0
    simulator.mujoco.mj_forward(simulator.model, simulator.data)
    for _ in range(400):
        simulator.mujoco.mj_step(simulator.model, simulator.data)


@pytest.fixture(scope="module")
def probe_cell():
    """Tare once, then measure every demo parcel with one plumb wrist reading."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, simplified_graphics=True)
    probe = ComProbe(simulator)
    tare = probe.calibrate()
    readings = []
    try:
        for index, package in enumerate(scenario.packages):
            if probe._held is not None:
                held = probe._held.index
                probe.release()
                parked = scenario.packages[held]
                simulator._set_package_pose(
                    held, (-3.0 - held * 0.7, 0.0, parked.size[2] / 2 + 0.002)
                )
            _reset_arm(simulator)
            cup = simulator.data.site_xpos[simulator.site_id].copy()
            simulator._set_package_pose(index, (cup[0], cup[1], cup[2] - package.size[2] / 2))
            probe.grasp(index)
            readings.append(probe.measure(index, tare))
    finally:
        if probe._held is not None:
            probe.release()
        simulator.close()
    return scenario, tare, readings


@pytest.fixture(scope="module")
def measured(probe_cell):
    scenario, tare, readings = probe_cell
    package = scenario.packages[HEAVY_INDEX]
    return scenario, package, tare, readings[HEAVY_INDEX]


def test_the_cell_declares_a_wrist_force_torque_sensor() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    model = mujoco.MjModel.from_xml_string(build_mjcf(scenario, simplified_graphics=True))
    assert model.sensor("ft_force").adr[0] >= 0
    assert model.sensor("ft_torque").adr[0] >= 0


def test_an_empty_tool_weighs_what_the_model_says() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    model = mujoco.MjModel.from_xml_string(build_mjcf(scenario, simplified_graphics=True))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    address = model.sensor("ft_force").adr[0]
    carried = float(np.linalg.norm(data.sensordata[address : address + 3]))
    assert carried == pytest.approx(PalletizingSimulator.tool_mass_kg * 9.81, rel=1e-3)


def test_the_probe_interpolates_wrist_poses_like_pick_and_place() -> None:
    """Wrist tilts must not step the joint command; pick-and-place uses a smoothstep."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, simplified_graphics=True)
    probe = ComProbe(simulator)
    start = np.asarray(simulator.data.qpos[simulator.arm_qpos], dtype=float).copy()
    target = start.copy()
    target[3:6] += MEASUREMENT_OFFSETS[1]
    commands = []
    original = simulator._capture_frame

    def capture(force: bool = False) -> None:
        commands.append(np.asarray(simulator.data.ctrl, dtype=float).copy())
        original(force)

    simulator._capture_frame = capture
    try:
        assert probe._settle(target)
        wrist_error = np.max(np.abs(simulator.data.qpos[simulator.arm_qpos][3:6] - target[3:6]))
        assert wrist_error < 0.01
    finally:
        simulator._capture_frame = original
        simulator.close()

    span = float(np.linalg.norm(target - start))
    jumps = [float(np.linalg.norm(commands[index + 1] - commands[index])) for index in range(len(commands) - 1)]
    assert len(commands) > 20
    assert np.linalg.norm(commands[0] - start) < 0.2 * span
    assert np.linalg.norm(commands[-1] - target) < 0.05
    assert max(jumps) < 0.25 * span


def test_a_carried_package_becomes_part_of_the_tool() -> None:
    """A sealed cup is a rigid joint; a weld equality would bias the measured torque."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    free = mujoco.MjModel.from_xml_string(build_mjcf(scenario, simplified_graphics=True))
    held = mujoco.MjModel.from_xml_string(
        build_mjcf(
            scenario,
            HeldPackage(HEAVY_INDEX, (0.0, 0.0, -0.2), (1.0, 0.0, 0.0, 0.0)),
            simplified_graphics=True,
        )
    )
    free_id = free.body(f"package_{HEAVY_INDEX}").id
    held_id = held.body(f"package_{HEAVY_INDEX}").id
    assert mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, held.body_parentid[held_id]) == "tool"
    assert held.body_jntnum[held_id] == 0
    assert held.neq == free.neq - 1  # its weld has nothing left to constrain
    # The viewer weld can be stiff without touching this path: there is no suction
    # equality left on a carried package for torquescale to act through.
    assert mujoco.mj_name2id(held, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{HEAVY_INDEX}") == -1
    # Re-parenting must not change what is being measured.
    assert held.body_mass[held_id] == pytest.approx(free.body_mass[free_id])
    assert np.allclose(held.body_ipos[held_id], free.body_ipos[free_id])
    assert np.allclose(held.body_inertia[held_id], free.body_inertia[free_id])


def test_the_tare_recovers_the_tool_on_the_real_cell(measured) -> None:
    _scenario, _package, tare, _measurement = measured
    assert tare.mass == pytest.approx(PalletizingSimulator.tool_mass_kg, abs=5e-3)
    assert tare.well_conditioned


def test_the_probe_recovers_mass_and_planar_centre_of_mass(measured) -> None:
    _scenario, package, _tare, measurement = measured
    assert measurement.trustworthy, measurement.estimate.reason
    assert measurement.mass == pytest.approx(package.mass, abs=5e-2)
    assert measurement.com_local[0] == pytest.approx(package.com[0], abs=2e-3)
    assert measurement.com_local[1] == pytest.approx(package.com[1], abs=2e-3)
    # The unseen component is the prior: the geometric centre, not the true height.
    assert measurement.com_local[2] == pytest.approx(0.0, abs=2e-3)
    assert measurement.hidden_m < 0.010
    assert measurement.method == "in_situ"


def test_in_situ_recovers_the_planar_centre_of_every_demo_package(probe_cell) -> None:
    """A plumb reading has to leave XY intact; the vertical is the geometric centre."""
    scenario, _tare, readings = probe_cell
    assert len(readings) == len(scenario.packages)
    for package, measurement in zip(scenario.packages, readings, strict=True):
        assert measurement.trustworthy, (package.id, measurement.estimate.reason)
        assert measurement.mass == pytest.approx(package.mass, abs=5e-2)
        assert measurement.com_local[0] == pytest.approx(package.com[0], abs=2e-3), package.id
        assert measurement.com_local[1] == pytest.approx(package.com[1], abs=2e-3), package.id
        assert measurement.com_local[2] == pytest.approx(0.0, abs=2e-3), package.id
        assert measurement.hidden_m < 0.010
        assert measurement.method == "in_situ"


def test_the_measurement_feeds_the_planner_model(measured) -> None:
    """`as_package` is the whole integration surface: it returns a planner-ready package."""
    _scenario, package, _tare, measurement = measured
    updated = measurement.as_package(package)
    assert updated.id == package.id
    assert updated.size == package.size
    assert np.allclose(updated.com[:2], package.com[:2], atol=2e-3)
    assert updated.com[2] == pytest.approx(0.0, abs=2e-3)
    # The off-centre load is real, not a rounding artefact.
    assert max(abs(value) for value in measurement.com_normalised[:2]) > 0.1


def test_the_cell_starts_without_knowing_any_centre_of_mass() -> None:
    """With measuring on, the planner may not peek at the scenario's centres of mass."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, measure_com=True, simplified_graphics=True)
    assert all(package.com == (0.0, 0.0, 0.0) for package in simulator.known_packages)
    assert any(package.com != (0.0, 0.0, 0.0) for package in scenario.packages)
    # Mass and size are taken as given: those come off the manifest, balance does not.
    assert [package.mass for package in simulator.known_packages] == [p.mass for p in scenario.packages]


def test_measuring_off_keeps_the_declared_centres_of_mass() -> None:
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, measure_com=False, simplified_graphics=True)
    assert [package.com for package in simulator.known_packages] == [p.com for p in scenario.packages]


def test_the_viewer_and_measuring_can_run_together() -> None:
    """Grasping reloads the same GLFW window instead of opening a second viewer."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, viewer=False, measure_com=True, simplified_graphics=True)
    try:
        simulator.rebuild(HeldPackage(HEAVY_INDEX, (0.0, 0.0, -0.2), (1.0, 0.0, 0.0, 0.0)))
        simulator.rebuild(None)
        assert simulator.model.body(f"package_{HEAVY_INDEX}").id >= 0
    finally:
        simulator.close()


def test_a_measurement_updates_what_the_cell_believes(measured) -> None:
    scenario, package, _tare, measurement = measured
    updated = measurement.as_package(package)
    assert updated.com[:2] != (0.0, 0.0)
    assert np.allclose(updated.com[:2], scenario.packages[HEAVY_INDEX].com[:2], atol=2e-3)


def test_precise_sweep_recovers_the_vertical_component_too() -> None:
    """The slow path still exists: four tilted poses resolve all three directions."""
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    simulator = PalletizingSimulator(scenario, simplified_graphics=True)
    probe = ComProbe(simulator, ProbeConfig(precise=True))
    tare = probe.calibrate()
    package = scenario.packages[HEAVY_INDEX]
    try:
        _reset_arm(simulator)
        cup = simulator.data.site_xpos[simulator.site_id].copy()
        simulator._set_package_pose(HEAVY_INDEX, (cup[0], cup[1], cup[2] - package.size[2] / 2))
        probe.grasp(HEAVY_INDEX)
        measurement = probe.measure(HEAVY_INDEX, tare)
    finally:
        if probe._held is not None:
            probe.release()
        simulator.close()
    assert measurement.trustworthy, measurement.estimate.reason
    assert measurement.method == "sweep"
    assert measurement.mass == pytest.approx(package.mass, abs=1e-2)
    assert np.allclose(measurement.com_local, package.com, atol=2e-3)
    assert measurement.estimate.rank == 3
