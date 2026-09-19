"""
Medición: el paquete ya está en la mano, ¿cómo de grande es, cuánto pesa, dónde tiene
el centro de gravedad?

Una lectura de muñeca con la herramienta a plomo. La componente que esa pose no ve es
la paralela a la gravedad —la altura del CoM dentro del cartón—, y ni el planificador
ni el margen al vuelco ni el CoG del palé la leen. Así que no se mide: se ancla en el
centro geométrico, las otras dos salen exactas, y un chivato comprueba que la dirección
ciega sigue siendo la que no importa.

La tara de la herramienta vacía sí barre varias orientaciones, una vez por episodio.
`--precise-com` deja el barrido también para cada bulto.

Esto ocurre DESPUÉS de coger y ANTES de planificar, y ése es el orden que define la
arquitectura: el planificador no puede precalcular el palé entero porque no sabe cómo
es el siguiente paquete hasta que lo tiene agarrado.

**El CoG se estima del par en la muñeca, no se lee del catálogo.** Un `cog_offset_m`
copiado de `configs/pallet.yaml` es un oráculo con otro nombre. Las dimensiones sí
vienen de lo que la percepción creyó al ver el bulto (`observation.dims_guess`); la
masa y el CoG salen del sensor. Las dos cosas están dichas aquí para que no se
confundan.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.contracts import Observation, PackageSpec
from tools.stable_pallet.com_estimator import (
    HIDDEN_LIMIT_M,
    TILTED,
    estimate_com,
    hidden_planar_error,
)
from tools.stable_pallet.tare import WRIST_OFFSETS, calibrate_tare
from tools.stable_pallet.wrench import average_wrench, read_wrench

# 30 muestras a 0,002 s: 0,06 s. El ruido blanco de la masa baja con √N; el plano
# horizontal lo manda la tara, no esta ventana. La tara, una vez por episodio, usa 120.
_AVERAGE_STEPS = 30
_TARE_AVERAGE_STEPS = 120
# Radians on shoulder_lift, only when the wrist is about to tilt.
_SWEEP_LIFT = 0.35


class WristGauge:
    """Medición con el paquete agarrado. Cumple `contracts.Gauge`."""

    def __init__(self, *, precise: bool = False):
        self.precise = precise
        self._tare = None

    def calibrate(self, scene, arm) -> None:
        """Pesa la herramienta vacía. Describe la herramienta, no el bulto, y se reutiliza."""
        if arm.held is not None:
            raise RuntimeError("calibrate: la herramienta está cogiendo un paquete")
        home = np.asarray(scene.home, dtype=float)
        self._tare = calibrate_tare(self._sweep(scene, arm, home, steps=_TARE_AVERAGE_STEPS))
        self._go_joints(scene, arm, home)

    def measure(self, scene, arm, observation: Observation) -> PackageSpec:
        if self._tare is None:
            raise RuntimeError("WristGauge.calibrate tiene que correr antes del primer pick")
        if arm.held is None:
            raise RuntimeError("measure: el paquete tiene que estar agarrado")
        if not arm.make_rigid():
            raise RuntimeError("measure: el agarre rígido no se sostuvo")

        box = next(item for item in scene.boxes if item.package_id == observation.package_id)
        held = scene.held
        prior = np.asarray(held.position, dtype=float)
        rotation = _quat_to_mat(scene, held.quaternion)
        half = np.asarray(observation.dims_guess, dtype=float) / 2.0
        resume = np.asarray(scene.data.qpos[scene.arm_qpos], dtype=float).copy()

        if self.precise:
            samples = self._tilted_samples(scene, arm, resume)
        else:
            samples = [self._sample(scene)]

        estimate = estimate_com(samples, self._tare, prior=prior)
        _lean, hidden = 0.0, 0.0
        if estimate.rank < 3:
            _lean, hidden = hidden_planar_error(estimate.blind_axis, rotation, half)
            if hidden > HIDDEN_LIMIT_M and not self.precise:
                samples = self._tilted_samples(scene, arm, resume)
                estimate = estimate_com(samples, self._tare, prior=prior)
                if estimate.rank < 3:
                    _lean, hidden = hidden_planar_error(estimate.blind_axis, rotation, half)
                else:
                    _lean, hidden = 0.0, 0.0
            if hidden > HIDDEN_LIMIT_M and estimate.reason == "ok":
                estimate = replace(estimate, reason=TILTED)

        self._go_joints(scene, arm, resume)
        com_local = rotation.T @ (estimate.com - prior)
        return PackageSpec(
            package_id=observation.package_id,
            type_name=box.type_name,
            dims_m=tuple(float(value) for value in observation.dims_guess),
            mass_kg=float(estimate.mass),
            cog_offset_m=np.asarray(com_local, dtype=float),
        )

    def _tilted_samples(self, scene, arm, resume: np.ndarray) -> list:
        """Levanta el hombro —solo aquí, porque va a inclinar— y barre la muñeca."""
        lifted = resume.copy()
        lifted[1] -= _SWEEP_LIFT
        self._go_joints(scene, arm, lifted)
        return self._sweep(scene, arm, np.asarray(scene.data.qpos[scene.arm_qpos], dtype=float))

    def _sweep(self, scene, arm, base: np.ndarray, *, steps: int = _AVERAGE_STEPS) -> list:
        samples = []
        for offset in WRIST_OFFSETS:
            target = np.asarray(base, dtype=float).copy()
            target[3:6] += offset
            self._go_joints(scene, arm, target)
            samples.append(self._sample(scene, steps=steps))
        return samples

    def _go_joints(self, scene, arm, target) -> None:
        target = np.asarray(target, dtype=float)
        if arm.fast_forward:
            arm._snap_to(target)
        else:
            arm.move_joints(target, 0.85)
        self._wait_static(scene)

    def _wait_static(self, scene) -> None:
        motion = scene.cfg["motion"]
        limit = float(motion["rest_speed"])
        dwell = int(motion["rest_steps"])
        still = 0
        for _ in range(int(motion["rest_timeout"])):
            if np.max(np.abs(scene.data.qvel[scene.arm_dofs])) < limit:
                still += 1
                if still >= dwell:
                    return
            else:
                still = 0
            _step_once(scene)

    def _sample(self, scene, *, steps: int = _AVERAGE_STEPS):
        self._wait_static(scene)
        batch = []
        for _ in range(steps):
            _step_once(scene)
            batch.append(read_wrench(scene.mujoco, scene.model, scene.data))
        return average_wrench(batch)


def _step_once(scene) -> None:
    scene.mujoco.mj_step(scene.model, scene.data)
    scene.clock += scene.model.opt.timestep
    scene.sync_viewer()


def _quat_to_mat(scene, quaternion) -> np.ndarray:
    rotation = np.empty(9)
    scene.mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    return rotation.reshape(3, 3)
