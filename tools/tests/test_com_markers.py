import math

import mujoco
import numpy as np
import pytest

from stable_pallet.com_markers import ComMarkers
from stable_pallet.pallet_com import com_from_world_points
from stable_pallet.scenario import load_scenario
from stable_pallet.simulator import COM_PACKAGE_ALPHA, PalletizingSimulator


class _StubViewer:
    """Just the part of the passive viewer the markers touch: a scene to fill."""

    def __init__(self, model) -> None:
        self.user_scn = mujoco.MjvScene(model, 200)

    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass


class _AppearanceViewer(_StubViewer):
    """Records package alpha values when the viewer copies the model."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model
        self.synced_alpha: list[np.ndarray] = []

    def sync(self) -> None:
        self.synced_alpha.append(self.model.geom_rgba[:, 3].copy())


def _cell(*, measure_com: bool = True) -> PalletizingSimulator:
    simulator = PalletizingSimulator(
        load_scenario("scenarios/mixed_boxes.yaml"), simplified_graphics=True, measure_com=measure_com
    )
    simulator.viewer = _StubViewer(simulator.model)
    simulator.com_markers = ComMarkers(simulator)
    return simulator


def _stage(simulator: PalletizingSimulator, indices: tuple[int, ...]) -> None:
    for index in indices:
        package = simulator.scenario.packages[index]
        simulator._set_package_pose(index, (0.2 + 0.4 * index, -0.1, 0.14 + package.size[2] / 2))


def test_nothing_is_drawn_until_a_marker_is_asked_for() -> None:
    simulator = _cell()
    try:
        _stage(simulator, (0, 1))
        simulator.com_markers.draw()
        assert simulator.viewer.user_scn.ngeom == 0
    finally:
        simulator.close()


def test_each_staged_package_gets_a_marker_plus_one_for_the_whole_load() -> None:
    simulator = _cell()
    try:
        _stage(simulator, (0, 1, 2))
        simulator.controls.show_true_com = True
        simulator.com_markers.draw()
        # Three packages, then the load marker: sphere, plumb line and the point where
        # the line meets the deck.
        assert simulator.viewer.user_scn.ngeom == 3 + 3

        simulator.controls.show_estimated_com = True
        simulator.planned_placements = []
        simulator.com_markers.draw()
        # No placement has been planned yet, so there is no load estimate to draw.
        assert simulator.viewer.user_scn.ngeom == 3 + 3 + 3
    finally:
        simulator.close()


def test_packages_are_translucent_only_while_com_markers_are_visible() -> None:
    simulator = _cell()
    try:
        _stage(simulator, (0,))
        viewer = _AppearanceViewer(simulator.model)
        simulator.viewer = viewer
        simulator.com_markers = ComMarkers(simulator)
        package_ids = [simulator._package_geom(index) for index in range(3)]
        original_alpha = simulator.model.geom_rgba[package_ids, 3].copy()

        simulator.controls.show_true_com = True
        simulator.sync_viewer()

        assert viewer.synced_alpha[-1][package_ids[0]] == pytest.approx(COM_PACKAGE_ALPHA)
        assert viewer.synced_alpha[-1][package_ids[1:]] == pytest.approx([0.0, 0.0])
        assert simulator.model.geom_rgba[package_ids, 3] == pytest.approx(
            viewer.synced_alpha[-1][package_ids]
        )
        assert simulator._semantic_model_rgba()[package_ids, 3] == pytest.approx(original_alpha)
        assert simulator._placed_package_indices() == [0]

        simulator.controls.show_true_com = False
        simulator.sync_viewer()

        assert viewer.synced_alpha[-1][package_ids] == pytest.approx(original_alpha)
        assert simulator.model.geom_rgba[package_ids, 3] == pytest.approx(original_alpha)
    finally:
        simulator.close()


def test_the_two_package_markers_are_apart_by_exactly_the_unknown_offset() -> None:
    """Before a carton is weighed the cell assumes it is balanced, and that is the error."""
    simulator = _cell(measure_com=True)
    try:
        index = next(
            i for i, package in enumerate(simulator.scenario.packages) if any(package.com)
        )
        _stage(simulator, (index,))
        markers = simulator.com_markers

        gap = markers._true_package_com(index) - markers._believed_package_com(index)

        declared = simulator.scenario.packages[index].com
        assert np.linalg.norm(gap) == pytest.approx(math.dist(declared, (0.0, 0.0, 0.0)), abs=1e-6)
    finally:
        simulator.close()


def test_weighing_a_carton_closes_the_gap_between_the_markers() -> None:
    simulator = _cell(measure_com=True)
    try:
        index = next(
            i for i, package in enumerate(simulator.scenario.packages) if any(package.com)
        )
        _stage(simulator, (index,))
        markers = simulator.com_markers
        simulator.known_packages[index] = simulator.scenario.packages[index]

        gap = markers._true_package_com(index) - markers._believed_package_com(index)

        assert np.linalg.norm(gap) == pytest.approx(0.0, abs=1e-9)
    finally:
        simulator.close()


def test_pallet_coordinates_land_where_the_pallet_actually_is() -> None:
    simulator = _cell()
    try:
        markers = simulator.com_markers
        point = (0.30, 0.25, 0.40)

        world = markers._pallet_to_world(point)

        simulation = simulator.scenario.simulation
        back = com_from_world_points(
            [1.0], [world], simulation.pallet_origin, simulation.pallet_height, simulator.scenario.pallet
        )
        assert back.com == pytest.approx(point, abs=1e-9)
    finally:
        simulator.close()


def test_the_estimated_marker_rides_the_pallet_through_a_jolt() -> None:
    """Both markers have to move with the deck, or the comparison stops meaning anything."""
    simulator = _cell()
    try:
        markers = simulator.com_markers
        before = markers._pallet_to_world((0.30, 0.25, 0.40))

        simulator.data.qpos[simulator.pallet_qpos[0]] += 0.05
        mujoco.mj_forward(simulator.model, simulator.data)
        after = markers._pallet_to_world((0.30, 0.25, 0.40))

        assert after - before == pytest.approx([0.05, 0.0, 0.0], abs=1e-9)
    finally:
        simulator.close()
