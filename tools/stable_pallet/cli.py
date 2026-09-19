"""La línea de comandos del demostrador. NO es el punto de entrada del repo.

El entrypoint es `scripts/palletize.py`, y lo es por un motivo concreto: es el único
sitio que elige las piezas —percepción de verdad o su stub-oráculo, heurística o
rejilla— y por tanto el único que puede calcular `oracle` del run. Si hubiera dos
caminos que abren episodios, tarde o temprano uno de los dos subiría runs con `oracle`
mal puesto, y la interfaz no compara un run con oráculo contra uno sin él: se
invalidaría justo la comparación que justifica el trabajo.

Así que esto **no abre episodios ni sube filas**. Sirve para ver la celda, depurar una
maniobra, probar una cámara, medir el planificador contra una línea base y lanzar el
panel de control. Todo lo que hace acaba en un JSON de `artifacts/`, nunca en Supabase.

Las banderas que sí aportaban —visor, velocidad, pausa, fast-forward, marcadores de
centro de masa, gráficos simplificados, vídeo— están también en `scripts/palletize.py`,
que es donde hay que usarlas si la ejecución tiene que contar.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .benchmark import run_benchmark, run_com_benchmark
from .controls import ViewerControls
from .experiments import EXPERIMENTS, RunOptions, experiment, run_experiment, summarise
from .generator import BoxGenerator, BoxGeneratorConfig
from .models import Package, StackState
from .planner import NoFeasiblePlacement, StablePalletPlanner
from .scenario import Scenario, dump_scenario, load_scenario
from .simulator import PalletizingSimulator, save_result
from .stability import validate_stack
from .trial import EVAL_BOX_COUNT, EVAL_SEED, run_generated_eval
from .truck import TruckBay


def _controls_from_args(args: argparse.Namespace) -> ViewerControls:
    return ViewerControls(
        speed=args.speed,
        fast_forward=args.fast_forward,
        show_true_com=args.show_com,
        show_estimated_com=args.show_estimated_com,
    )


def _print_plan(state: StackState) -> None:
    print("\nID             X      Y      Z   Yaw   Size (m)")
    print("------------- ------ ------ ------ ---- ----------------")
    for placement in state.placements:
        size = " x ".join(f"{value:.2f}" for value in placement.package.oriented_size(placement.yaw))
        print(
            f"{placement.package.id:<13} {placement.x:>6.3f} {placement.y:>6.3f} "
            f"{placement.z:>6.3f} {placement.yaw:>4} {size}"
        )


def command_plan(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    planner = StablePalletPlanner(scenario.planner)
    state = StackState(scenario.pallet)
    for index, _package in enumerate(scenario.packages):
        try:
            candidate = planner.plan_next(state, list(scenario.packages[index:]))
        except NoFeasiblePlacement as exc:
            print(str(exc), file=sys.stderr)
            return 2
        state = state.with_placement(candidate.placement)
    report = validate_stack(
        state,
        scenario.planner.minimum_support_ratio,
        scenario.planner.minimum_tipping_margin,
    )
    _print_plan(state)
    print(
        f"\nStable: {report.stable} | height: {state.max_height:.3f} m | "
        f"min support: {report.minimum_support_ratio:.1%} | "
        f"min tipping margin: {report.minimum_margin * 1000:.1f} mm"
    )
    if args.output:
        save_result(
            {
                "scenario": scenario.name,
                "success": report.stable,
                "minimum_support_ratio": report.minimum_support_ratio,
                "minimum_tipping_margin_m": report.minimum_margin,
                "state": state.as_dict(),
            },
            args.output,
        )
    return 0 if report.stable else 2


def _with_source(scenario: Scenario, source: str | None) -> Scenario:
    """Let the command line say where the cartons come from, whatever the file says."""
    if source is None:
        return scenario
    bay = (scenario.simulation.truck or TruckBay()) if source == "truck" else None
    return replace(scenario, simulation=replace(scenario.simulation, truck=bay))


def command_simulate(args: argparse.Namespace) -> int:
    scenario = _with_source(load_scenario(args.scenario), args.source)
    simulator = PalletizingSimulator(
        scenario,
        viewer=args.viewer,
        video_path=args.video,
        seed=args.seed,
        simplified_graphics=args.simplified_graphics,
        measure_com=args.measure_com,
        controls=_controls_from_args(args),
    )
    result = simulator.run()
    save_result(result, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "state"}, indent=2))
    if result.get("truck"):
        _print_unload(result["truck"])
    print(f"Full result: {Path(args.output).resolve()}")
    if args.video:
        print(f"Video: {Path(args.video).resolve()}")
    return 0 if result["success"] else 2


def _shake_from_args(args: argparse.Namespace, shake):
    updates = {}
    if args.duration is not None:
        updates["duration"] = args.duration
    return replace(shake, **updates) if updates else shake


def command_shake(args: argparse.Namespace) -> int:
    scenario = _with_source(load_scenario(args.scenario), args.source)
    simulator = PalletizingSimulator(
        scenario,
        viewer=args.viewer,
        video_path=args.video,
        seed=args.seed,
        simplified_graphics=args.simplified_graphics,
        measure_com=False if args.instant_place else args.measure_com,
        controls=_controls_from_args(args),
    )
    result = simulator.run(
        shake=_shake_from_args(args, scenario.simulation.shake),
        instant_place=args.instant_place,
    )
    save_result(result, args.output)
    printable = {key: value for key, value in result.items() if key != "state"}
    print(json.dumps(printable, indent=2))
    if result.get("truck"):
        _print_unload(result["truck"])
    print(f"Full result: {Path(args.output).resolve()}")
    if args.video:
        print(f"Video: {Path(args.video).resolve()}")
    if "shake" in result:
        _print_shake_scores(result["shake"])
    if "beam" in result:
        held = result["beam"]["summary"]["by_axis"]
        print(
            "Beam: "
            + ", ".join(f"{axis} {'holds' if stable else 'tips'}" for axis, stable in held.items())
        )
    return 0 if result["success"] else 2


def _print_shake_scores(shake: dict) -> None:
    summary = shake["summary"]
    print(
        "Transport sweep: "
        + ", ".join(
            f"{axis} holds to {info['max_held_g']:g} g"
            for axis, info in summary["by_axis"].items()
        )
    )
    trials = shake.get("trials") or []
    if not trials or "score" not in trials[0]:
        return
    axes = list(shake.get("axes") or [])
    levels: list[float] = []
    by_key = {(trial["peak_accel_g"], trial["axis"]): trial for trial in trials}
    for trial in trials:
        if trial["peak_accel_g"] not in levels:
            levels.append(trial["peak_accel_g"])
    print("\nShake scores (0-100, one per jolt):")
    print(f"{'g':>6} " + " ".join(f"{axis:>7}" for axis in axes))
    for level in levels:
        cells = []
        for axis in axes:
            trial = by_key[(level, axis)]
            cells.append(f"{trial['score']:7.1f}")
        print(f"{level:6.2f} " + " ".join(cells))
    print(
        f"\nmean {summary['mean_score']:.1f}  weighted {summary['weighted_score']:.1f}  "
        f"min {summary['min_score']:.1f}  held_all {summary['held_all']}"
    )


def _print_unload(truck: dict) -> None:
    print(
        f"\nTrailer: {len(truck['slots'])} cartons in {truck['columns']} columns, "
        f"stacked {truck['tallest_column_m'] * 1000:.0f} mm high"
    )
    print(f"{'carton':<12} {'col':>3} {'level':>5} {'top mm':>7} {'yaw':>6}")
    for slot in sorted(truck["slots"], key=lambda item: -item["top_m"]):
        print(
            f"{slot['package_id']:<12} {slot['column']:>3} {slot['level']:>5} "
            f"{slot['top_m'] * 1000:7.0f} {slot['yaw_deg']:6.1f}"
        )
    print("Picked: " + " -> ".join(truck["pick_order"]))
    print(
        f"Always the highest carton left: {truck['followed_plan']}  |  "
        f"rest of the load moved at most {truck['max_load_disturbance_mm']:.1f} mm"
    )


def _print_com_measurements(readings: list) -> None:
    print("\nWrist CoM (mm from geometric centre):")
    print(f"{'id':<12} {'kg':>6}  {'measured mm':<28} {'declared mm':<28} {'err':>6}")
    for item in readings:
        measured = ", ".join(f"{value:+6.1f}" for value in item["measured_com_mm"])
        declared = ", ".join(f"{value:+6.1f}" for value in item["declared_com_mm"])
        print(
            f"{item['package_id']:<12} {item['measured_mass_kg']:6.2f}  "
            f"({measured})  ({declared})  {item['error_mm']:6.1f}"
        )


def command_eval(args: argparse.Namespace) -> int:
    result = run_generated_eval(
        count=args.count,
        seed=args.seed,
        template=args.scenario,
        viewer=args.viewer,
        video_path=args.video,
        simplified_graphics=args.simplified_graphics,
        measure_com=False if args.instant_place else args.measure_com,
        instant_place=args.instant_place,
        name=args.name,
        controls=_controls_from_args(args),
        source=args.source or "truck",
    )
    save_result(result, args.output)
    print(
        f"Generated {result['generator']['count']} parcels (seed {result['generator']['seed']})"
    )
    print(
        f"Placed {result['placed']}/{result['generator']['count']}  "
        f"stable={result['success']}  height={result['state']['max_height']:.3f} m  "
        f"mode={result['placement_mode']}  source={result['source']}"
    )
    if result.get("truck"):
        _print_unload(result["truck"])
    if result.get("com_measurements"):
        _print_com_measurements(result["com_measurements"])
    print(f"Full result: {Path(args.output).resolve()}")
    if args.video:
        print(f"Video: {Path(args.video).resolve()}")
    if "shake" in result:
        _print_shake_scores(result["shake"])
    if "beam" in result:
        held = result["beam"]["summary"]["by_axis"]
        print(
            "Beam: "
            + ", ".join(f"{axis} {'holds' if stable else 'tips'}" for axis, stable in held.items())
        )
    return 0 if result["success"] and result["placed"] == result["generator"]["count"] else 2


def command_benchmark(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    result = run_benchmark(scenario, args.trials, args.seed)
    save_result(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"Full result: {Path(args.output).resolve()}")
    return 0


def command_benchmark_com(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    result = run_com_benchmark(scenario, args.trials, args.seed)
    save_result(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"Full result: {Path(args.output).resolve()}")
    return 0


def _print_generated(packages: tuple[Package, ...]) -> None:
    print("ID         L (m)  W (m)  H (m)   Vol (m³)   ρ (kg/m³)  Mass (kg)  CoM (m)")
    print("--------- ------ ------ ------ ---------- ---------- ---------- ----------------------")
    for package in packages:
        cx, cy, cz = package.com
        print(
            f"{package.id:<9} {package.size[0]:6.3f} {package.size[1]:6.3f} {package.size[2]:6.3f} "
            f"{package.volume:10.4f} {package.density:10.1f} {package.mass:10.3f} "
            f"({cx:+.4f}, {cy:+.4f}, {cz:+.4f})"
        )


def command_generate(args: argparse.Namespace) -> int:
    template = load_scenario(args.template)
    config = BoxGeneratorConfig(
        min_length=args.min_length,
        max_length=args.max_length,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        min_density=args.min_density,
        max_density=args.max_density,
        min_mass=args.min_mass,
        max_mass=args.max_mass,
        max_aspect_ratio=args.max_aspect_ratio,
    )
    packages = BoxGenerator(config, args.seed).generate_many(args.count)
    scenario = Scenario(
        args.name,
        template.pallet,
        packages,
        template.planner,
        template.simulation,
    )
    _print_generated(packages)
    dump_scenario(scenario, args.output)
    print(f"\nWrote {len(packages)} packages to {Path(args.output).resolve()}")
    return 0


def _add_viewer_args(parser: argparse.ArgumentParser) -> None:
    """Switches that only change what the viewer does, never what the cell computes."""
    parser.add_argument("--viewer", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wall-clock pacing: 1 runs one simulated second per second, 0 removes the pacing",
    )
    parser.add_argument(
        "--fast-forward",
        action="store_true",
        help="Jump the arm from waypoint to waypoint instead of driving the trajectories",
    )
    parser.add_argument(
        "--show-com",
        action="store_true",
        help="Draw the real centres of mass, the ones MuJoCo integrates",
    )
    parser.add_argument(
        "--show-estimated-com",
        action="store_true",
        help="Draw the centres of mass the cell has computed, which is what the planner used",
    )
    parser.add_argument(
        "--simplified-graphics",
        action="store_true",
        help="Use the original lightweight primitive graphics instead of the detailed cell",
    )


def _add_cell_args(parser: argparse.ArgumentParser, *, output: str) -> None:
    parser.add_argument("--scenario", default="scenarios/mixed_boxes.yaml")
    parser.add_argument("--output", default=output)
    parser.add_argument("--video", help="Optional MP4 output path")
    _add_viewer_args(parser)
    parser.add_argument(
        "--no-measure-com",
        dest="measure_com",
        action="store_false",
        help="Trust the centres of mass declared in the scenario instead of weighing each package",
    )
    parser.add_argument(
        "--source",
        choices=("truck", "conveyor"),
        default=None,
        help="Where the cartons come from: a loaded trailer the cell empties top carton "
        "first, or one singulated carton at a time on the infeed conveyor",
    )
    parser.add_argument("--seed", type=int, default=7)


def command_dashboard(args: argparse.Namespace) -> int:
    from .webapp import launch

    return launch(args.host, args.port, open_browser=args.browser)


def command_run(args: argparse.Namespace) -> int:
    item = experiment(args.experiment)
    result = run_experiment(
        item,
        RunOptions(
            viewer=args.viewer and item.watchable,
            controls=_controls_from_args(args),
            simplified_graphics=args.simplified_graphics,
            measure_com=args.measure_com,
            seed=args.seed,
            output=args.output,
        ),
    )
    print(item.title)
    for line in summarise(item, result):
        print(f"  {line}")
    print(f"Full result: {result['artifact']}")
    return 0 if result.get("success", True) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stable-pallet",
        description="Stable heterogeneous palletizing planner and MuJoCo demonstrator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Serve the control panel: pick an experiment, set the speed, pause and rewind",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8000)
    dashboard.add_argument(
        "--no-browser",
        dest="browser",
        action="store_false",
        help="Print the address instead of opening a browser",
    )
    dashboard.set_defaults(handler=command_dashboard)

    run = subparsers.add_parser(
        "run",
        help="Run one prepared experiment by name, with the same recipes the dashboard offers",
        description="Prepared experiments:\n"
        + "\n".join(f"  {item.key:<14} {item.title}" for item in EXPERIMENTS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("experiment", choices=[item.key for item in EXPERIMENTS])
    _add_viewer_args(run)
    run.add_argument("--output", help="Where to write the result (default: per experiment)")
    run.add_argument("--seed", type=int, help="Override the seed the experiment declares")
    run.add_argument(
        "--no-measure-com",
        dest="measure_com",
        action="store_false",
        default=None,
        help="Trust the declared centres of mass instead of weighing each package",
    )
    run.set_defaults(handler=command_run)

    plan = subparsers.add_parser("plan", help="Plan the complete sequence without physics")
    plan.add_argument("--scenario", default="scenarios/mixed_boxes.yaml")
    plan.add_argument("--output")
    plan.set_defaults(handler=command_plan)

    simulate = subparsers.add_parser("simulate", help="Run the closed-loop MuJoCo cell")
    _add_cell_args(simulate, output="artifacts/simulation.json")
    simulate.set_defaults(handler=command_simulate)

    shake = subparsers.add_parser(
        "shake",
        help="Palletize, then replay transport jolts and a narrow-beam stability check",
    )
    _add_cell_args(shake, output="artifacts/shake.json")
    shake.add_argument(
        "--instant-place",
        action="store_true",
        help="Teleport packages to the planned poses instead of running the robot, to preview the sweep",
    )
    shake.add_argument(
        "--duration",
        type=float,
        help="Duration of each one-cycle sine jolt, in seconds (default: scenario)",
    )
    shake.set_defaults(handler=command_shake)

    benchmark = subparsers.add_parser("benchmark", help="Compare lookahead against first-fit")
    benchmark.add_argument("--scenario", default="scenarios/mixed_boxes.yaml")
    benchmark.add_argument("--trials", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--output", default="artifacts/benchmark.json")
    benchmark.set_defaults(handler=command_benchmark)

    com = subparsers.add_parser(
        "benchmark-com",
        help="Compare the robot's pallet CoM estimate against MuJoCo truth",
    )
    com.add_argument("--scenario", default="scenarios/mixed_boxes.yaml")
    com.add_argument("--trials", type=int, default=10)
    com.add_argument("--seed", type=int, default=17)
    com.add_argument("--output", default="artifacts/pallet_com_benchmark.json")
    com.set_defaults(handler=command_benchmark_com)

    defaults = BoxGeneratorConfig()
    generate = subparsers.add_parser(
        "generate",
        help="Sample a random parcel set (size, density×volume mass, offset CoM) into a scenario YAML",
    )
    generate.add_argument("--count", type=int, default=8)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--name", default="random-parcels")
    generate.add_argument("--template", default="scenarios/mixed_boxes.yaml")
    generate.add_argument("--output", default="artifacts/random_boxes.yaml")
    generate.add_argument("--min-length", type=float, default=defaults.min_length)
    generate.add_argument("--max-length", type=float, default=defaults.max_length)
    generate.add_argument("--min-width", type=float, default=defaults.min_width)
    generate.add_argument("--max-width", type=float, default=defaults.max_width)
    generate.add_argument("--min-height", type=float, default=defaults.min_height)
    generate.add_argument("--max-height", type=float, default=defaults.max_height)
    generate.add_argument("--min-volume", type=float, default=defaults.min_volume)
    generate.add_argument("--max-volume", type=float, default=defaults.max_volume)
    generate.add_argument("--min-density", type=float, default=defaults.min_density)
    generate.add_argument("--max-density", type=float, default=defaults.max_density)
    generate.add_argument("--min-mass", type=float, default=defaults.min_mass)
    generate.add_argument("--max-mass", type=float, default=defaults.max_mass)
    generate.add_argument("--max-aspect-ratio", type=float, default=defaults.max_aspect_ratio)
    generate.set_defaults(handler=command_generate)

    evaluate = subparsers.add_parser(
        "eval",
        help="Generate random parcels, palletize with the robot, freeze the stack, shake and score",
    )
    _add_cell_args(evaluate, output="artifacts/eval.json")
    evaluate.add_argument("--count", type=int, default=EVAL_BOX_COUNT, help="Parcels to sample (fits a euro pallet)")
    evaluate.add_argument("--name", default=None, help="Scenario name written into the result")
    evaluate.add_argument(
        "--instant-place",
        action="store_true",
        help="Teleport packages to the planned poses instead of running the robot",
    )
    evaluate.set_defaults(handler=command_eval, seed=EVAL_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
