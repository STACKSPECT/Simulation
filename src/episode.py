"""Director del episodio: pide, ejecuta, mide y registra, en ese orden."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import measure
from src.cell.arm import ArmController
from src.cell.conveyor import make_supply
from src.cell.render import Snapshot, render
from src.contracts import Detector, Gauge, PackageSpec, PlacementPlan, Planner, Sink
from src.planner.naive import measured_heightmap


@dataclass
class Episode:
    """Medidas, eventos e imágenes producidos por una ejecución."""

    seed: int
    n_objects: int
    attempted: list = field(default_factory=list)
    placements: list[measure.Placement] = field(default_factory=list)
    final_placements: list[measure.Placement] = field(default_factory=list)
    states: list[measure.PalletState] = field(default_factory=list)
    drifts: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    snapshots: list[Snapshot] = field(default_factory=list)
    plans: list[PlacementPlan] = field(default_factory=list)
    failure: str | None = None
    duration_s: float = 0.0

    @property
    def n_placed(self) -> int:
        rows = self.final_placements or self.placements
        return sum(placement.placed for placement in rows)

    @property
    def success(self) -> bool:
        return self.failure is None and self.n_placed == self.n_objects


def run_episode(scene, detector: Detector, gauge: Gauge, planner: Planner,
                seed: int = 0, speed: float = 1.0, sink: Sink | None = None,
                verbose: bool = False) -> Episode:
    """Completa un episodio autónomo desde la fuente hasta el palé."""
    arm = ArmController(scene)
    arm.fast_forward = speed <= 0.0
    episode = Episode(seed=seed, n_objects=len(scene.boxes))
    scene.episode_specs = {}
    scene.episode_plans = {}
    supply = getattr(scene, "supply", None)
    if supply is None:
        supply = make_supply(scene)
        scene.supply = supply
        supply.stage(scene)
    max_duration = float(scene.cfg["episode"]["max_duration_s"])

    for _ in range(episode.n_objects):
        if scene.clock > max_duration:
            episode.failure = "timeout"
            break

        package_id = supply.present(scene)
        if package_id is None:
            episode.failure = None if supply.exhausted else "timeout"
            break
        observations = detector.observe(scene)
        source_state = supply._state(scene) if hasattr(supply, "_state") else {
            "source": scene.level.source
        }
        confidence = max((observation.confidence for observation in observations), default=0.0)
        _event(episode, scene, sink, "perceive", None,
               seen=len(observations), confidence=confidence, **source_state)
        if not observations:
            episode.failure = "no_detection"
            break

        observation = next(
            (item for item in observations if item.package_id == package_id), observations[0]
        )
        box = next(box for box in scene.boxes if box.package_id == observation.package_id)
        attempt = len(episode.attempted)
        episode.attempted.append(box)

        failure = _pick(arm, scene, box, observation)
        if failure:
            episode.failure = failure
            if arm.is_holding():
                _return_to_source(arm, scene, box, observation)
            _prepare_failed_attempt(scene, box, observation, attempt)
            _record(episode, scene, attempt, 0.0, sink, planner, verbose)
            break

        spec = gauge.measure(scene, arm, observation)
        scene.episode_specs[attempt] = spec
        _event(
            episode, scene, sink, "pick", box,
            mass_kg=round(spec.mass_kg, 4),
            type=spec.type_name,
            cog_offset_mm=[round(float(value) * 1000, 1) for value in spec.cog_offset_m],
        )

        heightmap = measured_heightmap(scene)
        centered = PackageSpec(
            package_id=spec.package_id,
            type_name=spec.type_name,
            dims_m=spec.dims_m,
            mass_kg=spec.mass_kg,
            cog_offset_m=np.zeros(3),
        )
        planner.choose(centered, heightmap)
        forecast = (
            gauge.forecast(scene, supply.forecast(scene))
            if hasattr(gauge, "forecast") else []
        )
        if hasattr(planner, "set_forecast"):
            planner.set_forecast(forecast)
        plan = planner.choose(spec, heightmap)
        if plan is None:
            episode.failure = "wrong_placement"
            _return_to_source(arm, scene, box, observation)
            _prepare_failed_attempt(scene, box, observation, attempt, spec)
            _record(episode, scene, attempt, 0.0, sink, planner, verbose)
            break

        scene.episode_plans[attempt] = plan
        episode.plans.append(plan)
        _event(
            episode, scene, sink, "plan", box,
            layer=plan.layer,
            slot=plan.slot,
            score=round(plan.score, 4),
            breakdown=plan.breakdown,
            heightmap_top_mm=round(heightmap.top * 1000, 1),
        )

        if len(episode.plans) > 1 and plan.layer > episode.plans[-2].layer:
            _shoot(episode, scene, arm, attempt - 1, sink)

        failure, drift = _place(arm, scene, box, spec, plan, episode.attempted)
        if failure and arm.is_holding():
            _return_to_source(arm, scene, box, observation)
        supply.release(scene)
        if failure:
            episode.failure = failure
        _record(episode, scene, attempt, drift, sink, planner, verbose)
        if episode.failure:
            break

    _finish(episode, scene, arm, sink)
    episode.duration_s = round(float(scene.clock), 2)
    if episode.failure:
        _event(episode, scene, sink, "fail", None, cause=episode.failure)
    return episode


def _prepare_failed_attempt(scene, box, observation, attempt: int,
                            spec: PackageSpec | None = None) -> None:
    """Da una fila medible a un intento que falló antes de recibir un plan."""
    if spec is None:
        spec = PackageSpec(
            box.package_id, box.type_name, tuple(observation.dims_guess), box.mass_kg,
            np.zeros(3),
        )
    px, py = scene.pallet_center
    scene.episode_specs[attempt] = spec
    scene.episode_plans[attempt] = PlacementPlan(
        position=np.array([px, py, scene.deck_z + spec.dims_m[2] / 2]),
        yaw=0.0,
        layer=1,
        slot="unplanned",
        score=0.0,
        predicted_support=0.0,
        breakdown={},
    )


def _pick(arm: ArmController, scene, box, observation) -> str | None:
    """Recoge el paquete presentado y lo eleva hasta la cota de tránsito."""
    motion = scene.cfg["motion"]
    top = np.asarray(observation.position, dtype=float).copy()
    top[2] += observation.dims_guess[2] / 2
    seal_z = float(top[2]) + arm.cup_gap
    safe_z = max(float(motion["transit_height"]), seal_z + float(motion["place_clearance"]))
    if not arm.go_to(top[0], top[1], safe_z, observation.yaw):
        return "ik_unreachable"
    if not arm.go_to(top[0], top[1], seal_z, observation.yaw, approach=True):
        return "ik_unreachable"
    grasp = arm.plan_grasp(box)
    if not arm.seal(box.index, grasp):
        return "grasp_slip"
    if not arm.go_to(top[0], top[1], safe_z, observation.yaw):
        return "ik_unreachable"
    return None if arm.is_holding() else "grasp_slip"


def _return_to_source(arm: ArmController, scene, box, observation) -> None:
    """Devuelve un paquete agarrado cuando el planificador no encuentra hueco."""
    if not arm.is_holding():
        return
    top = np.asarray(observation.position, dtype=float).copy()
    top[2] += observation.dims_guess[2] / 2
    safe_z = max(float(scene.cfg["motion"]["transit_height"]), float(top[2]) + 0.18)
    arm.go_to(top[0], top[1], safe_z, observation.yaw)
    arm.go_to(top[0], top[1], float(top[2]) + arm.cup_gap, observation.yaw, approach=True)
    arm.release()
    arm.go_to(top[0], top[1], safe_z, observation.yaw)


def _place(arm: ArmController, scene, box, spec: PackageSpec, plan: PlacementPlan,
           watched: list) -> tuple[str | None, float]:
    """Deposita, asienta, mide deriva y retira el brazo en cartesiano."""
    motion = scene.cfg["motion"]
    target_z = (
        float(plan.position[2]) + spec.dims_m[2] / 2 + arm.cup_gap
        + float(motion["drop_clearance"])
    )
    safe_z = max(
        float(motion["transit_height"]),
        target_z + float(motion["place_clearance"]),
        scene.deck_z + measured_heightmap(scene).top + float(motion["stack_clearance"]),
    )
    current = arm.tcp_pose().position
    if not arm.go_to(current[0], current[1], safe_z, plan.yaw):
        return "ik_unreachable", 0.0
    if not arm.go_to(plan.position[0], plan.position[1], safe_z, plan.yaw):
        return "ik_unreachable", 0.0
    if not arm.go_to(plan.position[0], plan.position[1], target_z, plan.yaw, approach=True):
        return "ik_unreachable", 0.0
    before = measure.positions(scene, watched)
    arm.release()
    scene.settle()
    drift = measure.max_displacement(before, measure.positions(scene, watched))
    if not arm.go_to(plan.position[0], plan.position[1], safe_z, plan.yaw):
        return "ik_unreachable", drift
    if arm.is_holding():
        return "grasp_slip", drift
    return None, drift


def _record(episode: Episode, scene, index: int, drift: float,
            sink: Sink | None, planner: Planner, verbose: bool) -> None:
    """Remide todas las cajas intentadas y emite una fila aunque haya fallo."""
    done: list[measure.Placement] = []
    for attempt, box in enumerate(episode.attempted):
        done.append(measure.measure_placement(scene, box, attempt, done))
    collapse = any(
        old.placed and not new.placed
        for old, new in zip(episode.placements, done)
    )
    episode.placements = done
    state = measure.pallet_state(measure.on_pallet(scene, done))
    episode.states.append(state)
    episode.drifts.append(drift)
    last = done[-1]
    _event(
        episode, scene, sink, "place", last.box,
        error_xy_mm=round(last.error_xy * 1000, 1),
        error_yaw_deg=round(float(np.degrees(last.error_yaw)), 1),
        overhang_mm=round(last.overhang * 1000, 1),
    )
    _event(
        episode, scene, sink, "settle", last.box,
        layer=last.layer, drift_mm=round(drift * 1000, 1),
    )
    if sink is not None:
        sink.placement(index, last)
        sink.pallet_state(index, state, drift)
    if hasattr(planner, "record") and measure.rect_overlap(last.rect, measure.pallet_rect(scene)):
        planner.record(last)
    if verbose:
        print(
            f"  {last.spec.package_id} {last.spec.type_name:<10} L{last.layer} "
            f"error {last.error_xy * 1000:5.1f} mm  apoyo {last.support_ratio:4.0%} "
            f"margen {state.stability_margin * 1000:+6.1f} mm  "
            f"{'ok' if last.placed else 'FALLO'}"
        )
    if collapse:
        episode.failure = "stack_collapse"
    elif episode.failure is None and not last.placed:
        episode.failure = (
            "overhang_violation"
            if last.overhang > float(scene.cfg["episode"]["max_overhang"])
            else "wrong_placement"
        )


def _shoot(episode: Episode, scene, arm: ArmController, after_seq: int,
           sink: Sink | None, *, final: bool = False) -> None:
    """Aparta el brazo en cartesiano, asienta y toma las vistas configuradas."""
    if final:
        arm.go_home()
    else:
        arm.park()
    scene.settle(0.25)
    for view in scene.cfg["episode"]["snapshot_views"]:
        shot = Snapshot(view=view, after_seq=after_seq, image=render(scene, view))
        episode.snapshots.append(shot)
        if sink is not None:
            sink.snapshot(shot)


def _finish(episode: Episode, scene, arm: ArmController, sink: Sink | None) -> None:
    """Retira el brazo, remide el resultado final y toma la última foto."""
    if not episode.attempted:
        episode.final_placements = []
        return
    final_seq = len(episode.attempted) - 1
    if not any(shot.after_seq == final_seq for shot in episode.snapshots):
        _shoot(episode, scene, arm, final_seq, sink, final=True)
    measured: list[measure.Placement] = []
    for index, box in enumerate(episode.attempted):
        measured.append(measure.measure_placement(scene, box, index, measured))
    if any(old.placed and not new.placed
           for old, new in zip(episode.placements, measured)):
        episode.failure = episode.failure or "stack_collapse"
    episode.final_placements = measured


def _event(episode: Episode, scene, sink: Sink | None, kind: str,
           box=None, **payload) -> None:
    row = {
        "ts": round(float(scene.clock), 2),
        "seq": len(episode.events),
        "kind": kind,
        "package_id": None if box is None else box.package_id,
        "payload": payload,
    }
    episode.events.append(row)
    if sink is not None:
        sink.event(row)
