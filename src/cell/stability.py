"""Ensayo físico de la pila que deja el robot: transporte y viga estrecha.

Portado del protocolo verificado de ``tools/stable_pallet/simulator.py``. Cada prueba
restaura un ``MjData`` tomado después del paletizado; las cajas nunca se sueldan al
palé, de modo que sólo el contacto y la fricción deciden si la carga aguanta.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

EURO_PALLET_MASS_KG = 25.0
GRAVITY = 9.81
TRANSPORT_REASONS = {
    0.05: "slow yard creep",
    0.15: "gentle vehicle acceleration",
    0.30: "typical forklift bump or urban braking",
    0.50: "EN 12195-1 lateral/rearward cargo securing",
    0.80: "EN 12195-1 forward securing (hard braking)",
}


def run_stability_test(
    scene,
    boxes: Iterable[Any],
    config: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Aplica 15 sacudidas y dos apoyos en viga sobre la pila física recibida."""
    load = list(boxes)
    if not load:
        result = {"ran": False, "reason": "no hay ninguna caja sobre el palé"}
        result["falls"] = count_fallen_boxes(result)
        return result
    protocol = dict(scene.cfg["stability_test"] if config is None else config)
    trial = _StabilityTrial(scene, load, protocol)
    result = {
        "ran": True,
        "shake": trial.run_transport(progress),
        "beam": trial.run_beam(progress),
    }
    result["falls"] = count_fallen_boxes(result)
    return result


def count_fallen_boxes(result: dict) -> dict:
    """Cuenta caídas caja-ensayo en las pruebas restauradas del protocolo.

    Cada sacudida y cada orientación de la viga parte de la misma pila. Por eso una
    misma caja que cae en dos ensayos cuenta dos veces en ``total``: son dos fallos de
    retención reproducibles, no una caja que siga en el suelo desde el ensayo anterior.
    ``unique_packages`` conserva además cuántos identificadores distintos fallaron.
    """
    if not result.get("ran"):
        return {
            "total": 0,
            "opportunities": 0,
            "unique_packages": 0,
            "package_ids": [],
            "by_protocol": {"shake": 0, "beam": 0},
        }

    total = 0
    opportunities = 0
    package_ids: set[str] = set()
    by_protocol: dict[str, int] = {}
    for protocol in ("shake", "beam"):
        protocol_total = 0
        for trial in result.get(protocol, {}).get("trials", []):
            packages = trial.get("packages", [])
            opportunities += len(packages)
            for package in packages:
                if not package.get("fell_off", False):
                    continue
                protocol_total += 1
                package_ids.add(str(package.get("package_id", "")))
        by_protocol[protocol] = protocol_total
        total += protocol_total
    package_ids.discard("")
    return {
        "total": total,
        "opportunities": opportunities,
        "unique_packages": len(package_ids),
        "package_ids": sorted(package_ids),
        "by_protocol": by_protocol,
    }


class _StabilityTrial:
    def __init__(self, scene, boxes: list[Any], config: dict) -> None:
        self.scene = scene
        self.boxes = boxes
        self.config = config
        self.baseline = None

    def run_transport(self, progress: Callable[[dict], None] | None = None) -> dict:
        cfg = self.config
        self._step(float(cfg["hold_seconds"]))
        self._capture_baseline()
        trials: list[dict] = []
        levels = tuple(float(value) for value in cfg["levels_g"])
        axes = tuple(str(value) for value in cfg["axes"])
        for level, peak_g in enumerate(levels, start=1):
            for axis in axes:
                self._restore_baseline()
                self._step(float(cfg["rest_seconds"]))
                result = self._apply_jolt(axis, peak_g)
                result["level"] = level
                result["score"] = _score(result, float(cfg["max_shift_m"]))
                trials.append(result)
                if progress is not None:
                    progress({
                        "phase": "stability",
                        "protocol": "shake",
                        "trial": len(trials),
                        "axis": axis,
                        "peak_accel_g": peak_g,
                        "fallen_boxes": sum(
                            bool(item["fell_off"]) for item in result["packages"]
                        ),
                        "held": bool(result["held"]),
                    })
        self._restore_baseline()
        return {
            "levels_g": list(levels),
            "axes": list(axes),
            "duration_s": float(cfg["duration"]),
            "max_shift_mm": float(cfg["max_shift_m"]) * 1_000,
            "trials": trials,
            "summary": _summarise_transport(trials, axes),
        }

    def run_beam(self, progress: Callable[[dict], None] | None = None) -> dict:
        cfg = self.config
        if self.baseline is None:
            self._step(float(cfg["hold_seconds"]))
            self._capture_baseline()
        trials = []
        for axis in ("x", "y"):
            self._restore_baseline()
            self._step(float(cfg["rest_seconds"]))
            result = self._apply_beam(axis)
            trials.append(result)
            if progress is not None:
                progress({
                    "phase": "stability",
                    "protocol": "beam",
                    "trial": len(trials),
                    "axis": axis,
                    "fallen_boxes": sum(
                        bool(item["fell_off"]) for item in result["packages"]
                    ),
                    "held": bool(result["held"]),
                })
        self._restore_baseline()
        return {
            "beam_width_mm": float(cfg["beam_width"]) * 1_000,
            "beam_height_mm": float(cfg["beam_height"]) * 1_000,
            "max_tilt_deg": float(cfg["max_tilt_deg"]),
            "trials": trials,
            "summary": {
                "held_all": all(item["held"] for item in trials),
                "trial_count": len(trials),
                "by_axis": {item["axis"]: item["held"] for item in trials},
            },
        }

    def _capture_baseline(self) -> None:
        self.baseline = self.scene.mujoco.MjData(self.scene.model)
        self.scene.mujoco.mj_copyData(self.baseline, self.scene.model, self.scene.data)

    def _restore_baseline(self) -> None:
        if self.baseline is None:
            raise RuntimeError("No se ha capturado la pila de referencia")
        clock = float(self.scene.data.time)
        self.scene.mujoco.mj_copyData(self.scene.data, self.scene.model, self.baseline)
        self.scene.data.time = clock
        self.scene.data.xfrc_applied[:] = 0
        self._configure_pallet(beam=False)
        self._park_beam()
        self.scene.sync_viewer()

    def _step(self, seconds: float, force: np.ndarray | None = None) -> None:
        count = max(1, round(seconds / self.scene.model.opt.timestep))
        for _ in range(count):
            if force is not None:
                self.scene.data.xfrc_applied[self.scene.pallet_body, :3] = force
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
            self.scene.clock += self.scene.model.opt.timestep
            self.scene.sync_viewer()

    def _relative_pose(self, box) -> tuple[np.ndarray, np.ndarray]:
        body = self.scene.body_id(box.index)
        pallet_pos = self.scene.data.xpos[self.scene.pallet_body]
        pallet_mat = self.scene.data.xmat[self.scene.pallet_body].reshape(3, 3)
        relative = pallet_mat.T @ (self.scene.data.xpos[body] - pallet_pos)
        inverse = np.empty(4)
        quaternion = np.empty(4)
        self.scene.mujoco.mju_negQuat(inverse, self.scene.data.xquat[self.scene.pallet_body])
        self.scene.mujoco.mju_mulQuat(
            quaternion, inverse, self.scene.data.xquat[body]
        )
        return relative, quaternion

    @staticmethod
    def _tilt_deg(quaternion: np.ndarray) -> float:
        _qw, qx, qy, _qz = quaternion
        value = max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))
        return math.degrees(math.acos(value))

    def _outcomes(self, before: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict], float, bool]:
        outcomes = []
        max_shift = 0.0
        any_fell = False
        max_allowed = float(self.config["max_shift_m"])
        width, depth = (float(value) for value in self.scene.pallet_dims)
        for box in self.boxes:
            relative, quaternion = self._relative_pose(box)
            previous, _ = before[box.index]
            shift = float(np.linalg.norm(relative - previous))
            fell = (
                abs(float(relative[0])) > width / 2
                or abs(float(relative[1])) > depth / 2
                or float(relative[2]) < self.scene.deck_z * 0.5
            )
            max_shift = max(max_shift, shift)
            any_fell = any_fell or fell
            outcomes.append({
                "package_id": box.package_id,
                "displacement_mm": shift * 1_000,
                "horizontal_shift_mm": float(np.linalg.norm((relative - previous)[:2])) * 1_000,
                "vertical_drop_mm": float(previous[2] - relative[2]) * 1_000,
                "tilt_deg": self._tilt_deg(quaternion),
                "fell_off": fell,
                "baseline_relative_center_m": [float(value) for value in previous],
                "relative_center_m": [float(value) for value in relative],
            })
        return outcomes, max_shift, (not any_fell) and max_shift <= max_allowed

    def _hover_force(self) -> np.ndarray:
        mass = EURO_PALLET_MASS_KG + sum(float(box.mass_kg) for box in self.boxes)
        return np.array([0.0, 0.0, mass * GRAVITY])

    def _apply_jolt(self, axis: str, peak_g: float) -> dict:
        cfg = self.config
        hover = self._hover_force()
        direction = _axis_unit(axis)
        total_mass = float(hover[2] / GRAVITY)
        peak_force = total_mass * peak_g * GRAVITY
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]

        self.scene.data.eq_active[self.scene.pallet_weld] = 0
        self._configure_pallet(beam=False)
        self.scene.data.xfrc_applied[self.scene.pallet_body, :3] = hover
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)
        self._step(0.08, hover)
        before = {box.index: self._relative_pose(box) for box in self.boxes}
        pallet_before = self.scene.data.xpos[self.scene.pallet_body].copy()

        duration = float(cfg["duration"])
        steps = max(1, round(duration / self.scene.model.opt.timestep))
        start = float(self.scene.data.time)
        measured_peak = 0.0
        previous_velocity = self.scene.data.qvel[self.scene.pallet_dofs].copy()
        for _ in range(steps):
            elapsed = float(self.scene.data.time) - start
            force = hover + peak_force * math.sin(2 * math.pi * elapsed / duration) * direction
            self.scene.data.xfrc_applied[self.scene.pallet_body, :3] = force
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
            self.scene.clock += self.scene.model.opt.timestep
            self.scene.sync_viewer()
            velocity = self.scene.data.qvel[self.scene.pallet_dofs].copy()
            acceleration = np.linalg.norm(
                (velocity - previous_velocity) / self.scene.model.opt.timestep
            )
            measured_peak = max(measured_peak, float(acceleration))
            previous_velocity = velocity
        self._step(float(cfg["settle_seconds"]), hover)
        self.scene.data.xfrc_applied[self.scene.pallet_body] = 0

        delta = self.scene.data.xpos[self.scene.pallet_body] - pallet_before
        outcomes, max_shift, held = self._outcomes(before)
        return {
            "axis": axis,
            "peak_accel_g": peak_g,
            "reason": TRANSPORT_REASONS.get(round(peak_g, 4), f"{peak_g:g} g transport jolt"),
            "peak_force_n": peak_force,
            "measured_peak_accel_g": measured_peak / GRAVITY,
            "duration_s": duration,
            "pallet_mass_kg": EURO_PALLET_MASS_KG,
            "loaded_mass_kg": total_mass,
            "pallet_travel_mm": float(np.linalg.norm(delta)) * 1_000,
            "axis_travel_mm": abs(float(delta[axis_index])) * 1_000,
            "max_package_shift_mm": max_shift * 1_000,
            "held": held,
            "packages": outcomes,
        }

    def _configure_pallet(self, *, beam: bool) -> None:
        for name in ("pallet_x", "pallet_y", "pallet_z"):
            joint = self._joint(name)
            self.scene.model.jnt_limited[joint] = 0
        for name in ("pallet_rx", "pallet_ry"):
            joint = self._joint(name)
            self.scene.model.jnt_limited[joint] = int(not beam)
            self.scene.model.jnt_range[joint] = (-0.0001, 0.0001) if not beam else (-1.6, 1.6)

    def _joint(self, name: str) -> int:
        return self.scene.mujoco.mj_name2id(
            self.scene.model, self.scene.mujoco.mjtObj.mjOBJ_JOINT, name
        )

    def _park_beam(self) -> None:
        mocap = self.scene.stability_beam_mocap
        self.scene.data.mocap_pos[mocap] = (0.0, 0.0, -1.0)
        self.scene.data.mocap_quat[mocap] = (1.0, 0.0, 0.0, 0.0)
        self.scene.model.geom_rgba[self.scene.stability_beam_geom, 3] = 0.0
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)

    def _stage_beam(self, axis: str) -> None:
        center = self.scene.data.xpos[self.scene.pallet_body]
        height = float(self.config["beam_height"])
        quaternion = (
            (1.0, 0.0, 0.0, 0.0)
            if axis == "x" else (0.70710678, 0.0, 0.0, 0.70710678)
        )
        mocap = self.scene.stability_beam_mocap
        self.scene.data.mocap_pos[mocap] = (float(center[0]), float(center[1]), height / 2)
        self.scene.data.mocap_quat[mocap] = quaternion
        self.scene.model.geom_rgba[self.scene.stability_beam_geom] = (0.78, 0.28, 0.14, 1.0)
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)

    def _lift_loaded_pallet(self) -> None:
        lift = float(self.config["beam_height"]) + 0.003
        self.scene.data.qpos[self.scene.pallet_qpos[2]] += lift
        self.scene.data.qvel[self.scene.pallet_dofs] = 0
        for box in self.boxes:
            joint = self.scene.mujoco.mj_name2id(
                self.scene.model, self.scene.mujoco.mjtObj.mjOBJ_JOINT, box.joint
            )
            if joint < 0:
                continue
            address = self.scene.model.jnt_qposadr[joint]
            dof = self.scene.model.jnt_dofadr[joint]
            self.scene.data.qpos[address + 2] += lift
            self.scene.data.qvel[dof:dof + 6] = 0.0
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)

    def _apply_beam(self, axis: str) -> dict:
        self.scene.data.eq_active[self.scene.pallet_weld] = 0
        self._lift_loaded_pallet()
        self._stage_beam(axis)
        self._configure_pallet(beam=True)
        self.scene.data.xfrc_applied[self.scene.pallet_body] = 0
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)
        before = {box.index: self._relative_pose(box) for box in self.boxes}
        self._step(float(self.config["beam_settle_seconds"]))

        matrix = self.scene.data.xmat[self.scene.pallet_body].reshape(3, 3)
        tilt_x = math.degrees(math.atan2(matrix[1, 2], matrix[2, 2]))
        tilt_y = math.degrees(math.atan2(-matrix[0, 2], matrix[2, 2]))
        pallet_z = float(self.scene.data.xpos[self.scene.pallet_body][2])
        outcomes, max_shift, _ = self._outcomes(before)
        limit = float(self.config["max_tilt_deg"])
        seated = pallet_z > float(self.config["beam_height"]) * 0.35
        held = (
            seated
            and not any(item["fell_off"] for item in outcomes)
            and abs(tilt_x) <= limit
            and abs(tilt_y) <= limit
        )
        return {
            "kind": "beam",
            "axis": axis,
            "reason": f"narrow beam along {axis}",
            "beam_width_mm": float(self.config["beam_width"]) * 1_000,
            "beam_height_mm": float(self.config["beam_height"]) * 1_000,
            "tilt_deg": tilt_x if axis == "x" else tilt_y,
            "tilt_x_deg": tilt_x,
            "tilt_y_deg": tilt_y,
            "max_tilt_deg": limit,
            "pallet_z_m": pallet_z,
            "seated": seated,
            "max_package_shift_mm": max_shift * 1_000,
            "held": held,
            "packages": outcomes,
        }


def _axis_unit(axis: str) -> np.ndarray:
    vectors = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    try:
        return vectors[axis]
    except KeyError:
        raise ValueError(f"Eje de sacudida desconocido: {axis!r}") from None


def _score(trial: dict, max_shift_m: float) -> float:
    # La misma guarda que `tools/stable_pallet/shake.py:93`: con 0 la fórmula divide
    # por cero y devuelve un score sin sentido en vez de avisar.
    if max_shift_m <= 0:
        raise ValueError("max_shift_m tiene que ser positivo")
    if any(item["fell_off"] for item in trial["packages"]):
        return 0.0
    worst = max((float(item["displacement_mm"]) for item in trial["packages"]), default=0.0) / 1_000
    return round(max(0.0, min(100.0, 100.0 * (1.0 - worst / (2 * max_shift_m)))), 1)


def _summarise_transport(trials: list[dict], axes: tuple[str, ...]) -> dict:
    by_axis = {}
    for axis in axes:
        rows = [item for item in trials if item["axis"] == axis]
        held = [item for item in rows if item["held"]]
        failed = [item for item in rows if not item["held"]]
        by_axis[axis] = {
            "max_held_g": max((item["peak_accel_g"] for item in held), default=0.0),
            "first_failure_g": failed[0]["peak_accel_g"] if failed else None,
            "mean_score": round(sum(item["score"] for item in rows) / len(rows), 1),
            "min_score": min(item["score"] for item in rows),
        }
    scores = [float(item["score"]) for item in trials]
    weights = [float(item["peak_accel_g"]) for item in trials]
    return {
        "held_all": all(item["held"] for item in trials),
        "trial_count": len(trials),
        "by_axis": by_axis,
        "mean_score": round(sum(scores) / len(scores), 1),
        "min_score": min(scores),
        "weighted_score": round(
            sum(score * weight for score, weight in zip(scores, weights, strict=True))
            / sum(weights),
            1,
        ),
    }
