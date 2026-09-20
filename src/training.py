"""Entrena los pesos de colocación con física de cajas, sin ejecutar el robot.

La política sigue siendo ``ScorePlanner``: sus catorce pesos deciden qué candidato
gana. El entrenador usa una búsqueda de política por entropía cruzada (CEM), que no
necesita gradientes ni otra dependencia. Cada evaluación coloca directamente los
paquetes en la pose elegida, deja actuar a MuJoCo y ejecuta el protocolo completo de
estabilidad sobre la pila resultante.

Esto es un banco de DECISIÓN, no un sustituto del episodio de celda: usa dimensiones,
masa, CoG y mapa de alturas oráculo, y elimina recogida, IK y ventosa a propósito.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from placing import METRIC_NAMES
from src import measure
from src.cell.scene import REPO, build_scene, load_configs
from src.cell.stability import count_fallen_boxes, run_stability_test
from src.contracts import PackageSpec
from src.planner.heightmap import measure_ground_truth
from src.planner.heuristic import ScorePlanner

Progress = Callable[[dict], None]


def resolve_weights(cfg: dict, values: Mapping[str, float] | None = None) -> dict[str, float]:
    """Mezcla pesos parciales con la configuración y valida los catorce términos."""
    weights = {
        str(name): float(value) for name, value in cfg["heuristic"]["weights"].items()
    }
    if values is not None:
        unknown = set(values) - set(METRIC_NAMES)
        if unknown:
            raise ValueError(f"pesos desconocidos: {sorted(unknown)}")
        weights.update({str(name): float(value) for name, value in values.items()})
    missing = set(METRIC_NAMES) - set(weights)
    if missing:
        raise ValueError(f"faltan pesos: {sorted(missing)}")
    for name in METRIC_NAMES:
        value = weights[name]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"el peso {name!r} tiene que ser finito y no negativo")
    if not any(weights[name] > 0.0 for name in METRIC_NAMES):
        raise ValueError("al menos un peso tiene que ser positivo")
    return {name: weights[name] for name in METRIC_NAMES}


def _trial_count(protocol: Mapping[str, Any]) -> int:
    return len(protocol["levels_g"]) * len(protocol["axes"]) + 2


@dataclass
class PlacementSimulationResult:
    """Resultado de una pila directa y su ensayo físico."""

    level: int
    seed: int
    weights: dict[str, float]
    n_objects: int
    n_planned: int
    n_placed: int
    n_on_pallet: int
    failure: str | None
    stability_trials: int
    decisions: list[dict]
    stability: dict
    simulated_seconds: float
    elapsed_seconds: float

    @property
    def n_missing(self) -> int:
        return self.n_objects - self.n_placed

    @property
    def falls(self) -> dict:
        return self.stability.get("falls") or count_fallen_boxes(self.stability)

    @property
    def loss(self) -> int:
        # Una caja que no se coloca no puede salir premiada por no caerse. La penaliza
        # un punto más que el peor caso posible de esa caja en todos los ensayos.
        return self.n_missing * (self.stability_trials + 1) + int(self.falls["total"])

    @property
    def reward(self) -> int:
        return -self.loss

    def summary(self) -> dict:
        return {
            "level": self.level,
            "seed": self.seed,
            "n_objects": self.n_objects,
            "n_planned": self.n_planned,
            "n_placed": self.n_placed,
            "n_on_pallet": self.n_on_pallet,
            "n_missing": self.n_missing,
            "failure": self.failure,
            "fallen_boxes_total": int(self.falls["total"]),
            "fall_opportunities": int(self.falls["opportunities"]),
            "unique_fallen_boxes": int(self.falls["unique_packages"]),
            "loss": self.loss,
            "reward": self.reward,
            "simulated_seconds": round(self.simulated_seconds, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    def detail(self) -> dict:
        return {
            **self.summary(),
            "weights": self.weights,
            "decisions": self.decisions,
            "stability": self.stability,
        }


def _measure_attempts(scene, boxes: list) -> list[measure.Placement]:
    placements: list[measure.Placement] = []
    for attempt, box in enumerate(boxes):
        placements.append(measure.measure_placement(scene, box, attempt, placements))
    return placements


def _decision(plan, box, index: int) -> dict:
    return {
        "seq": index,
        "package_id": box.package_id,
        "package_type": box.type_name,
        "planned_position_m": [float(value) for value in plan.position],
        "planned_yaw_rad": float(plan.yaw),
        "layer": int(plan.layer),
        "slot": plan.slot,
        "score": float(plan.score),
        "predicted_support": float(plan.predicted_support),
        "breakdown": {str(name): float(value) for name, value in plan.breakdown.items()},
    }


def simulate_placement(
    weights: Mapping[str, float] | None = None,
    *,
    level: int = 16,
    seed: int = 1,
    cfg: dict | None = None,
    stability_config: Mapping[str, Any] | None = None,
    progress: Progress | None = None,
) -> PlacementSimulationResult:
    """Coloca una carga sin brazo y devuelve caídas, recompensa y traza completa.

    Las cajas aparecen ``drop_clearance`` sobre el centro que el planificador eligió y
    caen por física, igual que al cortar el vacío. No se construye el UR10e, no se usa
    ``ArmController`` y no se monta ninguna fuente de suministro.
    """
    started = time.monotonic()
    source_cfg = load_configs() if cfg is None else cfg
    simulation_cfg = copy.deepcopy(source_cfg)
    selected_weights = resolve_weights(simulation_cfg, weights)
    simulation_cfg["heuristic"]["weights"] = dict(selected_weights)
    protocol = dict(simulation_cfg["stability_test"])
    if stability_config is not None:
        protocol.update(copy.deepcopy(dict(stability_config)))

    scene = build_scene(
        simulation_cfg,
        level_id=level,
        seed=seed,
        simplified=True,
        with_robot=False,
    )
    planner = ScorePlanner(simulation_cfg)
    scene.episode_specs = {}
    scene.episode_plans = {}
    attempted: list = []
    decisions: list[dict] = []
    placements: list[measure.Placement] = []
    failure: str | None = None
    drop = float(simulation_cfg["motion"]["drop_clearance"])

    try:
        for box in scene.boxes:
            spec = PackageSpec(
                package_id=box.package_id,
                type_name=box.type_name,
                dims_m=tuple(float(value) for value in box.dims_m),
                mass_kg=float(box.mass_kg),
                cog_offset_m=np.asarray(box.cog_offset_m, dtype=float),
            )
            heightmap = measure_ground_truth(scene)
            plan = planner.choose(spec, heightmap)
            if plan is None:
                failure = "no_candidate"
                decisions.append({
                    "seq": len(attempted),
                    "package_id": box.package_id,
                    "package_type": box.type_name,
                    "status": failure,
                    "reject": planner.last_reject,
                })
                if progress is not None:
                    progress({
                        "phase": "placement",
                        "seq": len(attempted),
                        "package_id": box.package_id,
                        "status": "no_candidate",
                        "reject": planner.last_reject,
                    })
                break

            attempt = len(attempted)
            attempted.append(box)
            scene.episode_specs[attempt] = spec
            scene.episode_plans[attempt] = plan
            decisions.append(_decision(plan, box, attempt))

            released = np.asarray(plan.position, dtype=float).copy()
            released[2] += drop
            scene.place_box(box.index, released, plan.yaw)
            scene.settle()

            previous = placements
            placements = _measure_attempts(scene, attempted)
            collapse = any(
                old.placed and not new.placed
                for old, new in zip(previous, placements)
            )
            measured = placements[-1]
            if measure.rect_overlap(measured.rect, measure.pallet_rect(scene)):
                planner.record(measured)
            if collapse:
                failure = "stack_collapse"
            elif not measured.placed:
                failure = "wrong_placement"

            if progress is not None:
                progress({
                    "phase": "placement",
                    "seq": attempt,
                    "package_id": box.package_id,
                    "status": "placed" if measured.placed else failure,
                    "layer": int(plan.layer),
                    "score": float(plan.score),
                    "error_xy_mm": float(measured.error_xy * 1_000),
                    "overhang_mm": float(measured.overhang * 1_000),
                })
            if failure is not None:
                break

        final_placements = _measure_attempts(scene, attempted)
        physical = measure.on_pallet(scene, final_placements)
        physical_boxes = [placement.box for placement in physical]
        stability = run_stability_test(
            scene, physical_boxes, protocol, progress=progress,
        )

        for decision, placement in zip(decisions, final_placements):
            decision.update({
                "actual_position_pallet_m": [float(value) for value in placement.position],
                "actual_yaw_rad": float(placement.yaw),
                "error_xy_mm": float(placement.error_xy * 1_000),
                "support_ratio": float(placement.support_ratio),
                "overhang_mm": float(placement.overhang * 1_000),
                "placed": bool(placement.placed),
                "status": "placed" if placement.placed else (failure or "not_placed"),
            })

        return PlacementSimulationResult(
            level=scene.level.id,
            seed=seed,
            weights=selected_weights,
            n_objects=len(scene.boxes),
            n_planned=len(attempted),
            n_placed=sum(bool(placement.placed) for placement in final_placements),
            n_on_pallet=len(physical),
            failure=failure,
            stability_trials=_trial_count(protocol),
            decisions=decisions,
            stability=stability,
            simulated_seconds=float(scene.clock),
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        scene.close()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class TrainingLog:
    """JSONL durable para progreso y JSON separados para evaluaciones completas."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.evaluations = directory / "evaluations"
        self.evaluations.mkdir(parents=True, exist_ok=False)
        self.progress_path = directory / "progress.jsonl"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def write(self, name: str, payload: dict) -> None:
        self._write_json(self.directory / name, payload)

    def evaluation(self, evaluation_id: str, payload: dict) -> str:
        relative = Path("evaluations") / f"{evaluation_id}.json"
        self._write_json(self.directory / relative, payload)
        return str(relative)

    def emit(self, event: str, **payload: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(row), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _repository_state(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *args), cwd=root, check=True, capture_output=True, text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _source_paths(root: Path) -> list[Path]:
    return [
        root / "scripts" / "train_weights.py",
        root / "src" / "contracts.py",
        root / "src" / "measure.py",
        root / "src" / "training.py",
        root / "src" / "planner" / "heuristic.py",
        root / "src" / "planner" / "heightmap.py",
        root / "src" / "cell" / "scene.py",
        root / "src" / "cell" / "stability.py",
        *sorted((root / "placing").glob("*.py")),
    ]


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _source_paths(root)
    }


def _snapshot_sources(destination: Path, root: Path) -> None:
    for source in _source_paths(root):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _runtime_versions() -> dict[str, str | None]:
    def version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": version("mujoco"),
        "PyYAML": version("PyYAML"),
    }


def _weight_id(weights: Mapping[str, float]) -> str:
    canonical = json.dumps(dict(weights), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _weights_from_logits(logits: np.ndarray, total: float) -> dict[str, float]:
    shifted = logits - float(np.max(logits))
    proportions = np.exp(shifted)
    proportions /= float(proportions.sum())
    return {
        name: float(total * proportions[index])
        for index, name in enumerate(METRIC_NAMES)
    }


def _default_run_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / "runs" / f"{stamp}-weight-training"


@dataclass(frozen=True)
class WeightTrainingResult:
    output_directory: Path
    best_weights: dict[str, float]
    best_loss: int
    best_missing: int
    best_fallen_boxes: int
    evaluations: int

    def summary(self) -> dict:
        return {
            "output_directory": str(self.output_directory),
            "best_weights": self.best_weights,
            "best_weight_id": _weight_id(self.best_weights),
            "best_loss": self.best_loss,
            "best_missing": self.best_missing,
            "best_fallen_boxes": self.best_fallen_boxes,
            "evaluations": self.evaluations,
        }


def train_weights(
    *,
    levels: Sequence[int] = (16,),
    seeds: Sequence[int] = (1,),
    iterations: int = 3,
    population: int = 4,
    elite_fraction: float = 0.5,
    sigma: float = 0.35,
    training_seed: int = 2026,
    cfg: dict | None = None,
    stability_config: Mapping[str, Any] | None = None,
    output_directory: Path | None = None,
    console: Callable[[str], None] | None = print,
    evaluator: Callable[..., PlacementSimulationResult] = simulate_placement,
) -> WeightTrainingResult:
    """Ajusta los pesos con CEM y registra cada paso antes de continuar.

    La clasificación minimiza ``missing * (n_trials + 1) + fallen_boxes``. Así una
    política no mejora simplemente aparcando cajas fuera; las caídas que una caja
    colocada provoca en el resto del montón siguen contando por separado.
    """
    if not levels or not seeds:
        raise ValueError("hay que indicar al menos un nivel y una semilla")
    if iterations < 1 or population < 2:
        raise ValueError("iterations debe ser >= 1 y population >= 2")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction tiene que estar en (0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma tiene que ser positivo")

    source_cfg = load_configs() if cfg is None else cfg
    base_cfg = copy.deepcopy(source_cfg)
    repository_root = Path(base_cfg.get("_root", REPO))
    initial = resolve_weights(base_cfg)
    protocol = dict(base_cfg["stability_test"])
    if stability_config is not None:
        protocol.update(copy.deepcopy(dict(stability_config)))

    output = Path(output_directory) if output_directory is not None else _default_run_directory(REPO)
    output.mkdir(parents=True, exist_ok=False)
    log = TrainingLog(output)
    config_snapshot = {
        str(key): value for key, value in base_cfg.items() if key != "_root"
    }
    config_json = json.dumps(
        _json_value(config_snapshot), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    log.write("config_snapshot.json", config_snapshot)
    _snapshot_sources(output / "source_snapshot", repository_root)
    manifest = {
        "schema_version": 1,
        "algorithm": "cross_entropy_policy_search",
        "objective": "missing * (stability_trials + 1) + fallen_boxes_total",
        "fallen_boxes_definition": (
            "suma de fell_off por caja y por ensayo restaurado; la misma caja puede "
            "contar en varios ensayos"
        ),
        "levels": [int(value) for value in levels],
        "seeds": [int(value) for value in seeds],
        "iterations": iterations,
        "population": population,
        "elite_fraction": elite_fraction,
        "sigma": sigma,
        "training_seed": training_seed,
        "initial_weights": initial,
        "stability_test": protocol,
        "config_snapshot": "config_snapshot.json",
        "config_sha256": config_sha256,
        "source_snapshot": "source_snapshot/",
        "source_sha256": _source_hashes(repository_root),
        "runtime_versions": _runtime_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_repository_state(repository_root),
    }
    log.write("manifest.json", manifest)
    log.emit("training_started", manifest="manifest.json")

    total_weight = float(sum(initial.values()))
    floor = total_weight * 1e-9
    mean = np.log(np.array([max(initial[name], floor) for name in METRIC_NAMES]))
    mean -= float(mean.mean())
    deviation = np.full(len(METRIC_NAMES), float(sigma))
    rng = np.random.default_rng(training_seed)
    elite_count = max(1, math.ceil(population * elite_fraction))
    global_best: dict | None = None
    evaluations = 0
    training_started = time.monotonic()

    for iteration in range(1, iterations + 1):
        candidates = [mean.copy()]
        candidates.extend(
            rng.normal(mean, deviation) for _ in range(population - 1)
        )
        rows: list[dict] = []
        log.emit("iteration_started", iteration=iteration, population=population)
        if console is not None:
            console(f"iteración {iteration}/{iterations} · {population} candidatos")

        for candidate_index, logits in enumerate(candidates, start=1):
            candidate_weights = _weights_from_logits(logits, total_weight)
            weight_id = _weight_id(candidate_weights)
            results: list[PlacementSimulationResult] = []
            log.emit(
                "candidate_started",
                iteration=iteration,
                candidate=candidate_index,
                weight_id=weight_id,
                weights=candidate_weights,
            )
            if console is not None:
                console(
                    f"  candidato {candidate_index}/{population} [{weight_id}] "
                    f"· {len(levels) * len(seeds)} evaluación(es)"
                )

            for level in levels:
                for seed in seeds:
                    evaluation_id = (
                        f"i{iteration:03d}-c{candidate_index:03d}"
                        f"-l{int(level):02d}-s{int(seed):06d}"
                    )
                    log.emit(
                        "evaluation_started",
                        evaluation_id=evaluation_id,
                        iteration=iteration,
                        candidate=candidate_index,
                        weight_id=weight_id,
                        level=int(level),
                        seed=int(seed),
                    )

                    def report(payload: dict, *, _evaluation_id=evaluation_id) -> None:
                        log.emit(
                            "evaluation_progress",
                            evaluation_id=_evaluation_id,
                            **payload,
                        )

                    try:
                        result = evaluator(
                            candidate_weights,
                            level=int(level),
                            seed=int(seed),
                            cfg=base_cfg,
                            stability_config=protocol,
                            progress=report,
                        )
                    except Exception as error:
                        log.emit(
                            "evaluation_failed",
                            evaluation_id=evaluation_id,
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                        raise
                    evaluations += 1
                    results.append(result)
                    detail_path = log.evaluation(evaluation_id, result.detail())
                    summary = result.summary()
                    log.emit(
                        "evaluation_finished",
                        evaluation_id=evaluation_id,
                        iteration=iteration,
                        candidate=candidate_index,
                        weight_id=weight_id,
                        detail=detail_path,
                        **summary,
                    )
                    if console is not None:
                        console(
                            f"    L{level} s{seed}: colocadas {result.n_placed}/"
                            f"{result.n_objects} · caídas {result.falls['total']}/"
                            f"{result.falls['opportunities']} · pérdida {result.loss} "
                            f"· {result.elapsed_seconds:.1f}s"
                        )

            row = {
                "iteration": iteration,
                "candidate": candidate_index,
                "weight_id": weight_id,
                "weights": candidate_weights,
                "logits": logits,
                "loss": sum(result.loss for result in results),
                "missing": sum(result.n_missing for result in results),
                "fallen_boxes": sum(int(result.falls["total"]) for result in results),
                "placed": sum(result.n_placed for result in results),
                "on_pallet": sum(result.n_on_pallet for result in results),
                "objects": sum(result.n_objects for result in results),
            }
            rows.append(row)
            log.emit(
                "candidate_finished",
                **{key: value for key, value in row.items() if key != "logits"},
            )
            key = (row["loss"], row["missing"], row["fallen_boxes"])
            if global_best is None or key < global_best["key"]:
                global_best = {**row, "key": key}

        rows.sort(key=lambda row: (row["loss"], row["missing"], row["fallen_boxes"]))
        elite = rows[:elite_count]
        elite_logits = np.stack([np.asarray(row["logits"], dtype=float) for row in elite])
        target_mean = elite_logits.mean(axis=0)
        target_deviation = elite_logits.std(axis=0)
        mean = 0.3 * mean + 0.7 * target_mean
        mean -= float(mean.mean())
        deviation = np.maximum(0.05, 0.5 * deviation + 0.5 * target_deviation)
        winner = rows[0]
        log.emit(
            "iteration_finished",
            iteration=iteration,
            best_weight_id=winner["weight_id"],
            best_loss=winner["loss"],
            best_missing=winner["missing"],
            best_fallen_boxes=winner["fallen_boxes"],
            elite_weight_ids=[row["weight_id"] for row in elite],
        )
        if console is not None:
            console(
                f"  mejor iteración [{winner['weight_id']}]: pérdida "
                f"{winner['loss']} · faltan {winner['missing']} · "
                f"caídas {winner['fallen_boxes']}"
            )

    assert global_best is not None
    result = WeightTrainingResult(
        output_directory=output,
        best_weights=dict(global_best["weights"]),
        best_loss=int(global_best["loss"]),
        best_missing=int(global_best["missing"]),
        best_fallen_boxes=int(global_best["fallen_boxes"]),
        evaluations=evaluations,
    )
    log.write("best_weights.json", {
        "weight_id": _weight_id(result.best_weights),
        "weights": result.best_weights,
    })
    summary = {
        **result.summary(),
        "elapsed_seconds": round(time.monotonic() - training_started, 3),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    log.write("summary.json", summary)
    log.emit("training_finished", summary="summary.json", **summary)
    if console is not None:
        console(
            f"terminado · mejor [{summary['best_weight_id']}] · "
            f"pérdida {result.best_loss} · logs {output}"
        )
    return result
