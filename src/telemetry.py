"""Única frontera entre la simulación y la plataforma de observabilidad."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
from theker_telemetry import EpisodeResult, RunLog

from src import measure

if TYPE_CHECKING:
    from src.cell.render import Snapshot
    from src.episode import Episode


def run_config(scene) -> dict:
    """Describe el montaje físico y la tarea en ``runs.config``."""
    pallet = scene.cfg["pallet"]
    source = scene.level.source
    return {
        "pallet_size_m": [_r(value) for value in pallet["dims"]],
        "pallet_deck_h_m": _r(pallet["deck_thickness"]),
        "pallet_scale": float(pallet["scale"]),
        "source": source,
        "level_name": scene.level.name,
        "robot": "Universal Robots UR10e",
        "eoat": scene.cfg["vacuum"]["name"],
        "catalogue": [
            {
                "package_id": box.package_id,
                "type": box.type_name,
                "dims_m": [_r(value) for value in box.dims_m],
                "mass_kg": _r(box.mass_kg),
            }
            for box in scene.boxes
        ],
        "motion": dict(scene.cfg["motion"]),
        "supply": dict(scene.cfg[source if source != "table" else "table"]),
        "auxiliary_table": dict(scene.cfg["auxiliary_table"]),
    }


class RunLogSink:
    """Convierte objetos de dominio en filas mientras el episodio está activo."""

    def __init__(self, log: RunLog, scene):
        self.log = log
        self.scene = scene

    def begin(self, seed: int) -> None:
        self.log.begin(seed, n_objects=len(self.scene.boxes))

    def end(self, episode: Episode) -> EpisodeResult:
        result = episode_result(episode, self.scene)
        self.log.end(result)
        return result

    def placement(self, index: int, placement: measure.Placement) -> None:
        self.log.placement(**placement_row(index, placement))

    def pallet_state(self, index: int, state: measure.PalletState, drift: float) -> None:
        self.log.pallet_state(**pallet_state_row(index, state, drift))

    def event(self, row: dict) -> None:
        self.log.event(**row)

    def snapshot(self, shot: Snapshot) -> None:
        self.log.snapshot(
            after_seq=shot.after_seq,
            view=shot.view,
            png=iio.imwrite("<bytes>", shot.image, extension=".png"),
            width=int(shot.image.shape[1]),
            height=int(shot.image.shape[0]),
        )


def placement_row(index: int, placement: measure.Placement) -> dict:
    """Convierte una caja intentada a las columnas de ``placements``."""
    return {
        "seq": int(index),
        "package_id": placement.spec.package_id,
        "package_type": placement.spec.type_name,
        "mass_kg": _r(placement.spec.mass_kg),
        "dims_m": [_r(value) for value in placement.spec.dims_m],
        "layer": int(placement.plan.layer),
        "planned_pose": _pose(placement.planned, placement.plan.yaw),
        "actual_pose": _pose(placement.position, placement.yaw),
        "error_xy_m": _r(placement.error_xy, 5),
        "error_yaw_rad": _r(placement.error_yaw, 5),
        "support_ratio": _r(placement.support_ratio),
        "overhang_m": _r(placement.overhang),
        "placed": bool(placement.placed),
    }


def pallet_state_row(index: int, state: measure.PalletState, drift: float) -> dict:
    """Convierte el estado posterior al intento a la traza de CoG."""
    return {
        "after_seq": int(index),
        "mass_kg": _r(state.mass_kg),
        "cog_x": _r(state.cog[0]),
        "cog_y": _r(state.cog[1]),
        "cog_z": _r(state.cog[2]),
        "stability_margin_m": _r(state.stability_margin),
        "fill_ratio": _r(state.fill_ratio),
        "settle_drift_m": _r(drift),
    }


def episode_result(episode: Episode, scene) -> EpisodeResult:
    """Resume el episodio y agrega los scores del planificador."""
    state = episode.states[-1] if episode.states else None
    physical = measure.on_pallet(scene, episode.final_placements)
    scores = [plan.score for plan in episode.plans]
    names = {name for plan in episode.plans for name in plan.breakdown}
    breakdown = {
        name: _r(np.mean([plan.breakdown.get(name, 0.0) for plan in episode.plans]))
        for name in sorted(names)
    }
    metrics = {
        "cog_offset_xy": _r(np.linalg.norm(state.cog[:2])) if state else 0.0,
        "fill_ratio": _r(state.fill_ratio) if state else 0.0,
        "settle_drift": _r(max(episode.drifts)) if episode.drifts else 0.0,
        "n_layers": max((placement.layer for placement in physical), default=0),
        "layer_flatness": _r(measure.layer_flatness(physical)),
        "max_overhang": _r(
            max((placement.overhang for placement in physical), default=0.0)
        ),
        "mean_planner_score": _r(np.mean(scores)) if scores else 0.0,
        "planner_score_breakdown": breakdown,
        "score": round(
            sum(placement.placed for placement in episode.final_placements)
            / episode.n_objects,
            4,
        ) if episode.n_objects else 0.0,
    }
    # El ensayo no es una columna: va dentro de `metrics`, que es `jsonb` y donde
    # sobrar es inocuo. Sólo el resumen; los 51 KB de detalle van a `stability.json`.
    if episode.stability_test is not None:
        test = episode.stability_test
        metrics["stability_test"] = (
            {
                "ran": True,
                "shake": test["shake"]["summary"],
                "beam": test["beam"]["summary"],
            }
            if test.get("ran") else test
        )
    return EpisodeResult(
        seed=episode.seed,
        level=scene.level.id,
        n_objects=episode.n_objects,
        n_placed=sum(placement.placed for placement in episode.final_placements),
        n_misrouted=0,
        success=episode.success,
        duration_s=episode.duration_s,
        failure=episode.failure,
        oracle=bool(getattr(scene, "oracle", False)),
        # La tarea es de dónde se coge: mesa, cinta o camión. Sale de la fuente del
        # nivel y no de una constante, que es lo que permite comparar las tres.
        task=scene.level.source,
        metrics=metrics,
    )


def save_snapshots(episode: Episode, directory: Path) -> dict[str, Path]:
    """Guarda todas las imágenes en disco, aunque no haya telemetría remota."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for shot in episode.snapshots:
        path = directory / f"{shot.after_seq:03d}-{shot.view}.png"
        iio.imwrite(path, shot.image)
        paths[path.stem] = path
    return paths


def _pose(position, yaw: float) -> dict:
    return {
        "x": _r(position[0]),
        "y": _r(position[1]),
        "z": _r(position[2]),
        "yaw": _r(yaw),
    }


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)
