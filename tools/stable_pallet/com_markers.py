"""Centre-of-mass markers drawn on top of the passive viewer.

Two things are worth seeing side by side while the cell works: where the mass actually
is, which only MuJoCo knows, and where the cell believes it is, which is the number the
planner acted on. The gap between the two markers is the whole argument for weighing a
package on the wrist instead of trusting a label, and it shrinks in front of you as each
carton is measured.

The markers live in the viewer's `user_scn`, so they are pure overlay: nothing here
touches the model, the data or the result of the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .pallet_com import estimate_pallet_com

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .simulator import PalletizingSimulator

TRUE_COLOR = (0.16, 0.92, 0.45, 0.95)
ESTIMATED_COLOR = (1.00, 0.46, 0.10, 0.95)
ERROR_COLOR = (0.98, 0.86, 0.22, 0.85)

PACKAGE_RADIUS = 0.018
LOAD_RADIUS = 0.034
PLUMB_WIDTH = 0.004
ERROR_WIDTH = 0.006


class ComMarkers:
    """Populates `viewer.user_scn` with the measured and computed centres of mass."""

    def __init__(self, simulator: PalletizingSimulator) -> None:
        self.sim = simulator
        self._identity = np.eye(3, dtype=float).flatten()

    def draw(self) -> None:
        sim = self.sim
        if sim.viewer is None:
            return
        scene = sim.viewer.user_scn
        if scene is None:
            return
        scene.ngeom = 0
        controls = sim.controls
        if not controls.draws_markers:
            return

        indices = sim._placed_package_indices()
        if controls.show_true_com:
            for index in indices:
                self._sphere(scene, self._true_package_com(index), PACKAGE_RADIUS, TRUE_COLOR)
        if controls.show_estimated_com:
            for index in indices:
                self._sphere(scene, self._believed_package_com(index), PACKAGE_RADIUS, ESTIMATED_COLOR)

        truth = self._true_load_com(indices) if controls.show_true_com else None
        estimate = self._estimated_load_com() if controls.show_estimated_com else None
        if truth is not None:
            self._load_marker(scene, truth, TRUE_COLOR, "CoM real")
        if estimate is not None:
            self._load_marker(scene, estimate, ESTIMATED_COLOR, "CoM calculado")
        if truth is not None and estimate is not None:
            self._line(scene, estimate, truth, ERROR_WIDTH, ERROR_COLOR)

    # -- the four points ------------------------------------------------------------

    def _true_package_com(self, index: int) -> np.ndarray:
        """Where MuJoCo integrates the package's weight, in world coordinates."""
        return self.sim.data.xipos[self._body(index)].copy()

    def _believed_package_com(self, index: int) -> np.ndarray:
        """The same point rebuilt from what the cell thinks the package is.

        Until the wrist weighs a carton the belief is "centred", so this marker sits at
        the geometric centre and the offset to the green one is the error the planner is
        working with.
        """
        sim = self.sim
        body = self._body(index)
        rotation = sim.data.xmat[body].reshape(3, 3)
        com = np.asarray(sim.known_packages[index].com, dtype=float)
        return sim.data.xpos[body] + rotation @ com

    def _true_load_com(self, indices: list[int]) -> np.ndarray | None:
        sim = self.sim
        mass = 0.0
        moment = np.zeros(3, dtype=float)
        for index in indices:
            body = self._body(index)
            body_mass = float(sim.model.body_mass[body])
            mass += body_mass
            moment += body_mass * sim.data.xipos[body]
        return moment / mass if mass > 0.0 else None

    def _estimated_load_com(self) -> np.ndarray | None:
        """The planner's own estimate, carried onto the pallet wherever the pallet is.

        Routing it through the pallet body rather than the static origin keeps the marker
        glued to the deck through a transport jolt, so the two markers stay comparable
        while the whole stack is moving.
        """
        placements = self.sim.planned_placements
        if not placements:
            return None
        estimate = estimate_pallet_com(placements, self.sim.scenario.pallet)
        return self._pallet_to_world(estimate.com)

    def _pallet_to_world(self, point: tuple[float, float, float]) -> np.ndarray:
        sim = self.sim
        pallet = sim.scenario.pallet
        local = np.array(
            [
                point[0] - pallet.width / 2,
                point[1] - pallet.depth / 2,
                point[2] + sim.scenario.simulation.pallet_height,
            ]
        )
        rotation = sim.data.xmat[sim.pallet_body_id].reshape(3, 3)
        return sim.data.xpos[sim.pallet_body_id] + rotation @ local

    def _body(self, index: int) -> int:
        return self.sim.mujoco.mj_name2id(self.sim.model, self.sim.mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")

    # -- drawing --------------------------------------------------------------------

    def _load_marker(self, scene: Any, position: np.ndarray, color: tuple[float, ...], label: str) -> None:
        """A sphere at the load centre of mass, plus the plumb line down to the deck."""
        self._sphere(scene, position, LOAD_RADIUS, color, label=label)
        deck = self._pallet_to_world((0.0, 0.0, 0.0))
        normal = self.sim.data.xmat[self.sim.pallet_body_id].reshape(3, 3)[:, 2]
        foot = position - normal * float(np.dot(position - deck, normal))
        self._line(scene, foot, position, PLUMB_WIDTH, color)
        self._sphere(scene, foot, PACKAGE_RADIUS * 0.8, color)

    def _sphere(
        self,
        scene: Any,
        position: np.ndarray,
        radius: float,
        color: tuple[float, ...],
        *,
        label: str = "",
    ) -> None:
        geom = self._next(scene)
        if geom is None:
            return
        self.sim.mujoco.mjv_initGeom(
            geom,
            self.sim.mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray(position, dtype=float),
            self._identity,
            np.asarray(color, dtype=np.float32),
        )
        geom.label = label

    def _line(
        self,
        scene: Any,
        start: np.ndarray,
        end: np.ndarray,
        width: float,
        color: tuple[float, ...],
    ) -> None:
        geom = self._next(scene)
        if geom is None:
            return
        mujoco = self.sim.mujoco
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            self._identity,
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
        )
        geom.label = ""

    @staticmethod
    def _next(scene: Any) -> Any | None:
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom
