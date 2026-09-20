"""The prepared experiments, in one place.

Each entry is a named recipe -- which scenario, who places the cartons, what happens to
the pallet afterwards -- so the dashboard and the command line offer the same list
instead of each growing its own set of flag combinations. Adding an experiment means
adding a row to `EXPERIMENTS`, not a new subcommand and a new panel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .benchmark import plan_stack, run_benchmark, run_com_benchmark
from .controls import ViewerControls
from .models import StackState
from .planner import StablePalletPlanner
from .scenario import Scenario, load_scenario
from .simulator import PalletizingSimulator, save_result
from .stability import validate_stack
from .trial import EVAL_BOX_COUNT, EVAL_SEED, run_generated_eval
from .truck import TruckBay

DEFAULT_SCENARIO = "scenarios/mixed_boxes.yaml"


@dataclass(frozen=True, slots=True)
class Experiment:
    """One prepared run: what to load, who places the cartons, what to do afterwards."""

    key: str
    title: str
    description: str
    kind: str = "cell"  # cell | generated | plan | benchmark | com-benchmark
    scenario: str = DEFAULT_SCENARIO
    # Where the cartons come from. The conveyor hands them over one at a time in the
    # order the scenario declares; the trailer hands over the whole load at once and
    # the cell works the order out for itself.
    source: str = "conveyor"
    shake: bool = False
    instant_place: bool = False
    measure_com: bool = True
    seed: int = 7
    count: int = EVAL_BOX_COUNT
    trials: int = 20
    output: str = "artifacts/run.json"

    @property
    def watchable(self) -> bool:
        """Whether there is a MuJoCo cell to open the viewer on."""
        return self.kind in {"cell", "generated"}

    @property
    def uses_robot(self) -> bool:
        """Whether the UR10e does the picking, rather than the cartons being teleported."""
        return self.watchable and not self.instant_place


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        key="palletize",
        title="Paletizado en lazo cerrado",
        description=(
            "El UR10e pesa cada caja en la muñeca, la paletiza y replanifica con la pose medida. "
            "Ocho paquetes de e-commerce, sin ensayo de transporte."
        ),
        shake=False,
        output="artifacts/simulation.json",
    ),
    Experiment(
        key="transport",
        title="Paletizado + ensayo de transporte",
        description=(
            "Lo anterior y, con el palé cargado tal como lo dejó el robot, 15 sacudidas "
            "(0,05–0,80 g en X, Y y Z) y el asiento sobre viga estrecha en X y en Y."
        ),
        shake=True,
        output="artifacts/shake.json",
    ),
    Experiment(
        key="pallet-test",
        title="Solo el ensayo del palé",
        description=(
            "Salta el brazo: el planificador coloca la pila de golpe y se pasa directamente "
            "a las 15 sacudidas y a la viga. Es la vía rápida para mirar la estabilidad."
        ),
        shake=True,
        instant_place=True,
        output="artifacts/pallet_test.json",
    ),
    Experiment(
        key="eval",
        title="Test-run completo (cajas aleatorias)",
        description=(
            "Genera seis paquetes nuevos, los pesa, los paletiza, congela el palé y puntúa "
            "cada sacudida de 0 a 100. Es la prueba de extremo a extremo."
        ),
        kind="generated",
        shake=True,
        seed=EVAL_SEED,
        count=EVAL_BOX_COUNT,
        output="artifacts/eval.json",
    ),
    Experiment(
        key="truck-unload",
        title="Descarga de camión y paletizado",
        description=(
            "Las ocho cajas llegan apiladas en el remolque, no de una en una por la cinta. "
            "La celda decide el orden sola: siempre la cara más alta que queda libre. "
            "Sin báscula de muñeca: esta tarjeta enseña la descarga, el pesaje va en la otra."
        ),
        source="truck",
        shake=False,
        measure_com=False,
        seed=EVAL_SEED,
        output="artifacts/truck_unload.json",
    ),
    Experiment(
        key="truck-eval",
        title="Test-run completo desde camión",
        description=(
            "Seis paquetes nuevos cargados en el remolque, descargados, pesados y paletizados, "
            "y después las 15 sacudidas. Es la prueba de extremo a extremo con camión."
        ),
        kind="generated",
        source="truck",
        shake=True,
        seed=EVAL_SEED,
        count=EVAL_BOX_COUNT,
        output="artifacts/truck_eval.json",
    ),
    Experiment(
        key="adversarial",
        title="Centros de masa adversarios",
        description=(
            "Escenario con centros de masa muy desplazados, colocado de golpe. Enseña que el "
            "planificador no confunde 'mucha superficie apoyada' con estabilidad."
        ),
        scenario="scenarios/adversarial_com.yaml",
        instant_place=True,
        output="artifacts/adversarial.json",
    ),
    Experiment(
        key="plan",
        title="Solo planificación (sin física)",
        description="Resuelve la secuencia completa con beam search y valida la pila. Instantáneo.",
        kind="plan",
        output="artifacts/plan.json",
    ),
    Experiment(
        key="benchmark",
        title="Benchmark: lookahead contra first-fit",
        description="20 órdenes aleatorios de las mismas cajas, los dos planificadores, misma semilla.",
        kind="benchmark",
        seed=17,
        trials=20,
        output="artifacts/benchmark.json",
    ),
    Experiment(
        key="benchmark-com",
        title="Benchmark del centro de masas del palé",
        description="Compara el CoM que estima el robot con el que mide MuJoCo, sobre 10 pilas.",
        kind="com-benchmark",
        seed=17,
        trials=10,
        output="artifacts/pallet_com_benchmark.json",
    ),
)

EXPERIMENTS_BY_KEY: dict[str, Experiment] = {item.key: item for item in EXPERIMENTS}


def experiment(key: str) -> Experiment:
    try:
        return EXPERIMENTS_BY_KEY[key]
    except KeyError:
        known = ", ".join(EXPERIMENTS_BY_KEY)
        raise KeyError(f"Unknown experiment {key!r}; pick one of: {known}") from None


@dataclass(frozen=True, slots=True)
class RunOptions:
    """How to run an experiment, as opposed to which one to run."""

    viewer: bool = False
    controls: ViewerControls | None = None
    simplified_graphics: bool = False
    measure_com: bool | None = None
    precise_com: bool = False
    seed: int | None = None
    count: int | None = None
    trials: int | None = None
    video_path: str | Path | None = None
    output: str | Path | None = None


def run_experiment(item: Experiment, options: RunOptions | None = None) -> dict[str, Any]:
    """Run one prepared experiment and write its artifact."""
    options = options or RunOptions()
    runners = {
        "cell": _run_cell,
        "generated": _run_generated,
        "plan": _run_plan,
        "benchmark": _run_benchmark,
        "com-benchmark": _run_com_benchmark,
    }
    try:
        runner = runners[item.kind]
    except KeyError:
        raise ValueError(f"Unknown experiment kind {item.kind!r}") from None
    result = runner(item, options)
    result["experiment"] = item.key
    destination = Path(options.output or item.output)
    save_result(result, destination)
    result["artifact"] = str(destination.resolve())
    return result


def _measures_com(item: Experiment, options: RunOptions) -> bool:
    if item.instant_place:
        return False
    return item.measure_com if options.measure_com is None else options.measure_com


def _from_source(scenario: Scenario, source: str) -> Scenario:
    """The same cell, fed by a trailer or by the infeed conveyor.

    A scenario file may declare its own bay; asking for a trailer without one gets the
    default. It is the presence of the bay that switches the cell over, so asking for
    the conveyor is a matter of taking it away again.
    """
    if source not in {"truck", "conveyor"}:
        raise ValueError(f"source must be 'truck' or 'conveyor', not {source!r}")
    bay = (scenario.simulation.truck or TruckBay()) if source == "truck" else None
    if bay is scenario.simulation.truck:
        return scenario
    return replace(scenario, simulation=replace(scenario.simulation, truck=bay))


def _run_cell(item: Experiment, options: RunOptions) -> dict[str, Any]:
    scenario = _from_source(load_scenario(item.scenario), item.source)
    simulator = PalletizingSimulator(
        scenario,
        viewer=options.viewer,
        video_path=options.video_path,
        seed=options.seed if options.seed is not None else item.seed,
        simplified_graphics=options.simplified_graphics,
        measure_com=_measures_com(item, options),
        precise_com=options.precise_com and _measures_com(item, options),
        controls=options.controls,
    )
    result = simulator.run(shake=item.shake, instant_place=item.instant_place)
    result["placed"] = len(result["state"]["placements"])
    result["expected"] = len(scenario.packages)
    return result


def _run_generated(item: Experiment, options: RunOptions) -> dict[str, Any]:
    result = run_generated_eval(
        count=options.count if options.count is not None else item.count,
        seed=options.seed if options.seed is not None else item.seed,
        template=item.scenario,
        viewer=options.viewer,
        video_path=options.video_path,
        simplified_graphics=options.simplified_graphics,
        measure_com=_measures_com(item, options),
        precise_com=options.precise_com and _measures_com(item, options),
        instant_place=item.instant_place,
        shake=item.shake,
        controls=options.controls,
        source=item.source,
    )
    result["expected"] = result["generator"]["count"]
    return result


def _run_plan(item: Experiment, options: RunOptions) -> dict[str, Any]:
    scenario = load_scenario(item.scenario)
    placements = plan_stack(scenario, list(scenario.packages), StablePalletPlanner(scenario.planner))
    state = StackState(scenario.pallet, placements)
    report = validate_stack(
        state,
        scenario.planner.minimum_support_ratio,
        scenario.planner.minimum_tipping_margin,
    )
    return {
        "scenario": scenario.name,
        "success": report.stable,
        "placed": len(placements),
        "expected": len(scenario.packages),
        "stability_reason": report.reason,
        "minimum_support_ratio": report.minimum_support_ratio,
        "minimum_tipping_margin_m": report.minimum_margin,
        "state": state.as_dict(),
    }


def _run_benchmark(item: Experiment, options: RunOptions) -> dict[str, Any]:
    scenario = load_scenario(item.scenario)
    trials = options.trials if options.trials is not None else item.trials
    seed = options.seed if options.seed is not None else item.seed
    return run_benchmark(scenario, trials, seed)


def _run_com_benchmark(item: Experiment, options: RunOptions) -> dict[str, Any]:
    scenario = load_scenario(item.scenario)
    trials = options.trials if options.trials is not None else item.trials
    seed = options.seed if options.seed is not None else item.seed
    return run_com_benchmark(scenario, trials, seed)


def summarise(item: Experiment, result: dict[str, Any]) -> list[str]:
    """A handful of lines a dashboard can print without knowing what ran."""
    if item.kind == "benchmark":
        return [
            f"{method}: completa {values['completion_rate']:.0%}, "
            f"estable {values['strict_stability_rate']:.0%}, "
            f"CoM descentrado {values['mean_com_offset_m'] * 1_000:.0f} mm"
            for method, values in result["summary"].items()
        ]
    if item.kind == "com-benchmark":
        summary = result["summary"]
        return [
            f"Error del CoM estimado: medio {summary['mean_error_norm_mm']:.1f} mm, "
            f"máximo {summary['max_error_norm_mm']:.1f} mm",
            f"Error de masa medio: {summary['mean_mass_error_kg']:.3f} kg",
        ]

    lines = [
        f"Colocadas {result['placed']}/{result['expected']} · "
        f"pila {'estable' if result['success'] else 'INESTABLE'} · "
        f"altura {result['state']['max_height']:.3f} m"
    ]
    if not result["success"] and result.get("stability_reason"):
        lines.append(f"Motivo: {result['stability_reason']}")
    lines.append(
        f"Apoyo mínimo {result['minimum_support_ratio']:.0%} · "
        f"margen al vuelco {result['minimum_tipping_margin_m'] * 1_000:.0f} mm"
    )

    truck = result.get("truck")
    if truck:
        lines.append(
            f"Remolque: {len(truck['slots'])} cajas en {truck['columns']} columnas, "
            f"apiladas {truck['tallest_column_m'] * 1_000:.0f} mm"
        )
        lines.append(
            f"Siempre la caja más alta: {'sí' if truck['followed_plan'] else 'no'} · "
            f"el resto de la carga se movió {truck['max_load_disturbance_mm']:.1f} mm como mucho"
        )

    readings = result.get("com_measurements") or []
    if readings:
        mean_error = sum(item["error_mm"] for item in readings) / len(readings)
        mean_xy = sum(item.get("error_xy_mm", item["error_mm"]) for item in readings) / len(readings)
        replanned = sum(1 for item in readings if item["replanned"])
        lines.append(
            f"Pesaje en muñeca: error medio {mean_error:.1f} mm "
            f"({mean_xy:.2f} mm en planta) en {len(readings)} cajas, "
            f"{replanned} replanificadas tras medir"
        )
    pallet_com = result.get("pallet_com")
    if pallet_com:
        lines.append(
            f"CoM del palé: calculado vs real {pallet_com['error_norm_mm']:.1f} mm "
            f"({pallet_com['error_xy_mm']:.1f} mm en planta)"
        )
    shake = result.get("shake")
    if shake:
        summary = shake["summary"]
        lines.append(
            f"Sacudidas: media {summary['mean_score']:.1f}/100 · mínima {summary['min_score']:.1f} · "
            f"aguanta las 15: {'sí' if summary['held_all'] else 'no'}"
        )
    beam = result.get("beam")
    if beam:
        held = beam["summary"]["by_axis"]
        lines.append(
            "Viga estrecha: "
            + ", ".join(f"{axis} {'aguanta' if stable else 'vuelca'}" for axis, stable in held.items())
        )
    return lines
