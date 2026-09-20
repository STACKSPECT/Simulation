#!/usr/bin/env python3
"""Ejecuta la celda autónoma con una de las tres fuentes y nueve niveles."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402
from theker_telemetry import EpisodeResult, RunLog  # noqa: E402

from src import measure  # noqa: E402
from src.cell.render import draw_heightmap  # noqa: E402
from src.cell.scene import build_scene, levels, load_configs  # noqa: E402
from src.cell.stability import run_stability_test  # noqa: E402
from src.episode import run_episode  # noqa: E402
from src.planner.heuristic import ScorePlanner  # noqa: E402
from src.planner.naive import BeamPlanner, GridPlanner  # noqa: E402
from src.telemetry import (  # noqa: E402
    RunLogSink,
    episode_result,
    run_config,
    save_snapshots,
)
from src.vision.detect import CameraDetector  # noqa: E402
from src.vision.gauge import WristGauge  # noqa: E402
from src.vision.oracle import OracleDetector, OracleGauge  # noqa: E402


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("-n", "--episodes", type=int, default=1)
    cli.add_argument("--seed", type=int, default=1)
    cli.add_argument("--source", choices=("table", "conveyor", "truck"))
    cli.add_argument("--level", type=int)
    cli.add_argument("--viewer", action="store_true")
    cli.add_argument("--speed", type=float,
                     help="segundos simulados por segundo real en el visor; "
                          "0 = fast-forward. Por defecto 1 con visor y 0 sin visor")
    cli.add_argument("--pause", type=float, default=0.0,
                     help="segundos de pausa entre episodios")
    cli.add_argument("--show-com", action="store_true",
                     help="incluye el CoG final en el informe")
    cli.add_argument("--simplified-graphics", action="store_true")
    cli.add_argument("--video", type=Path,
                     help="guarda un timelapse MP4 con las vistas cenitales")
    cli.add_argument("--protocol", choices=("text", "json"), default="text")
    cli.add_argument("--no-telemetry", action="store_true")
    cli.add_argument("--label", default="UR10e · tres fuentes")
    cli.add_argument("--oracle-vision", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--oracle-gauge", action=argparse.BooleanOptionalAction, default=True)
    # El mapa de alturas: de las poses de MuJoCo, o de las tres cámaras de percepción.
    cli.add_argument("--oracle-heightmap", action=argparse.BooleanOptionalAction,
                     default=True)
    cli.add_argument(
        "--precise-com",
        action="store_true",
        help="inclina la muñeca en varias poses en vez de una lectura a plomo",
    )
    cli.add_argument("--naive-planner", action="store_true",
                     help="relleno en rejilla: la línea base, y marca el run como oráculo")
    cli.add_argument("--beam-planner", action="store_true",
                     help="el beam search de tools/stable_pallet, para comparar contra él")
    cli.add_argument(
        "--stability-test",
        action="store_true",
        help="al terminar, aplica 15 sacudidas de transporte y la viga estrecha",
    )
    return cli


def _emit(args, payload: dict) -> None:
    if args.protocol == "json":
        print(json.dumps(payload), flush=True)
    elif payload["kind"] == "log":
        print(payload["text"], flush=True)


def _log(args, text: str) -> None:
    _emit(args, {"kind": "log", "text": text})


def _select_level(cli: argparse.ArgumentParser, args, cfg: dict) -> int:
    catalogue = levels(cfg)
    if args.level is None and args.source is None:
        return int(cfg["episode"]["level"])
    if args.level is None:
        return min(level.id for level in catalogue.values() if level.source == args.source)
    if args.level not in catalogue:
        cli.error(f"nivel {args.level} desconocido; disponibles: {sorted(catalogue)}")
    if args.source and catalogue[args.level].source != args.source:
        cli.error(
            f"el nivel {args.level} es de {catalogue[args.level].source}, no de {args.source}"
        )
    return args.level


def _parts(args, cfg: dict):
    detector = OracleDetector() if args.oracle_vision else CameraDetector()
    gauge = OracleGauge() if args.oracle_gauge else WristGauge(precise=args.precise_com)
    # La heurística de score es el DEFECTO. Los otros dos existen para compararse contra
    # ella: `GridPlanner` es la línea base tonta y `BeamPlanner` el beam search medido.
    if args.naive_planner:
        planner = GridPlanner(cfg)
    elif args.beam_planner:
        planner = BeamPlanner(cfg)
    else:
        planner = ScorePlanner(cfg)
    oracle = oracle_for(args.oracle_vision, args.oracle_gauge, args.naive_planner,
                        args.oracle_heightmap)
    return detector, gauge, planner, oracle


def oracle_for(oracle_vision: bool, oracle_gauge: bool, naive_planner: bool,
               oracle_heightmap: bool = True) -> bool:
    """El run es oráculo si cualquiera de sus piezas es un stub.

    El mapa de alturas cuenta como una pieza más: leerlo de las poses de MuJoCo es hacer
    trampa igual que leer el catálogo en vez de medir el paquete. Tiene bandera propia
    porque es un sensor distinto del de la estación de recogida —uno mira el palé y el
    otro la cinta—, y porque `vision/detect.py` sigue sin implementar: sin separarlos, el
    mapa medido con cámaras sería código inalcanzable.
    """
    return any((oracle_vision, oracle_gauge, naive_planner, oracle_heightmap))


def _run(scene, detector, gauge, planner, seed: int, speed: float, sink, args):
    def execute():
        episode = run_episode(
            scene, detector, gauge, planner, seed=seed, speed=speed,
            sink=sink, verbose=args.protocol == "text",
        )
        if args.stability_test:
            _log(args, "estabilidad: ensayando 15 sacudidas y viga estrecha…")
            physical = measure.on_pallet(scene, episode.final_placements)
            episode.stability_test = run_stability_test(
                scene, [placement.box for placement in physical]
            )
            for line in _stability_lines(episode.stability_test):
                _log(args, line)
        return episode

    if not args.viewer:
        return execute()
    import mujoco.viewer

    def on_key(keycode: int) -> None:
        # `m` enciende y apaga la rejilla del mapa de alturas. GLFW manda la letra en
        # mayúscula, así que se comparan las dos y no dependemos de si hay bloq mayús.
        if keycode not in (ord("m"), ord("M")):
            return
        scene.show_heightmap = not getattr(scene, "show_heightmap", False)
        last = getattr(scene, "last_heightmap", None)
        if last is not None:
            draw_heightmap(scene, last)

    with mujoco.viewer.launch_passive(
        scene.model, scene.data, key_callback=on_key
    ) as viewer:
        scene.viewer = viewer
        try:
            return execute()
        finally:
            scene.viewer = None


def _request_unwind(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def install_termination_unwind() -> None:
    """SIGTERM (p. ej. ``Popen.terminate``) tiene que deshacer igual que Ctrl-C."""
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_unwind)


def _aborted(seed: int, scene) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        level=scene.level.id,
        n_objects=len(scene.boxes),
        n_placed=0,
        n_misrouted=0,
        success=False,
        duration_s=round(float(scene.clock), 2),
        failure=None,
        oracle=bool(getattr(scene, "oracle", False)),
        task=scene.level.source,
        metrics={"aborted": True},
    )


def _save_video(path: Path, episode, index: int, total: int) -> None:
    frames = [shot.image for shot in episode.snapshots if shot.view == "top"]
    if not frames:
        return
    destination = path if total == 1 else path.with_stem(f"{path.stem}-{index:03d}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(destination, frames, fps=1, codec="libx264")


def _stability_lines(result: dict) -> list[str]:
    if not result.get("ran"):
        return [f"estabilidad: no ejecutada · {result.get('reason', 'sin carga')}"]
    shake = result["shake"]["summary"]
    beam = result["beam"]["summary"]
    axes = ", ".join(
        f"{axis} {'aguanta' if held else 'vuelca'}"
        for axis, held in beam["by_axis"].items()
    )
    return [
        (
            f"estabilidad · sacudidas {shake['mean_score']:.1f}/100 · "
            f"mínima {shake['min_score']:.1f} · aguanta las 15: "
            f"{'sí' if shake['held_all'] else 'no'}"
        ),
        f"estabilidad · viga estrecha: {axes}",
    ]


def _save_stability(directory: Path, result: dict | None) -> None:
    if result is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stability.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.episodes < 1:
        cli.error("--episodes debe ser positivo")
    if args.viewer and args.episodes != 1:
        cli.error("--viewer solo admite un episodio")
    speed = args.speed if args.speed is not None else (1.0 if args.viewer else 0.0)
    if speed < 0:
        cli.error("--speed no puede ser negativo")

    cfg = load_configs(REPO)
    level_id = _select_level(cli, args, cfg)
    first_scene = build_scene(
        cfg, level_id=level_id, seed=args.seed, simplified=args.simplified_graphics
    )
    _, _, _, oracle = _parts(args, cfg)
    first_scene.oracle = oracle
    # De dónde sale el mapa de alturas: con esto puesto, de las poses de MuJoCo; sin
    # ello, de las tres cámaras de percepción.
    first_scene.oracle_heightmap = args.oracle_heightmap
    log = RunLog(
        REPO,
        task=first_scene.level.source,
        level=level_id,
        oracle=oracle,
        motion_speed=1.0 if speed == 0.0 else speed,
        tag="pallet",
        label=args.label,
        n_episodes=args.episodes,
        config=run_config(first_scene),
        remote=not args.no_telemetry,
    )
    if args.no_telemetry:
        _log(args, "telemetría: solo disco (--no-telemetry)")
    elif log.run_id:
        _log(args, f"telemetría: ACTIVA · {log.path_in_ui}")
    else:
        _log(args, "telemetría: solo disco · faltan credenciales SUPABASE")
    # Qué planificador corre va en la línea de arranque por el mismo motivo que la
    # telemetría: dos ejecuciones que se comparan entre sí tienen que poder distinguirse
    # mirando la salida, no recordando qué banderas se pusieron.
    _, _, chosen, _ = _parts(args, cfg)
    _log(args, f"modo: {'oráculo' if oracle else 'sin oráculo'} · "
               f"{type(chosen).__name__} · "
               f"mapa {'oráculo' if args.oracle_heightmap else 'de cámaras'} · "
               f"{first_scene.level.name} · {args.episodes} episodio(s)")

    ok = 0
    seed = args.seed
    scene = first_scene
    lines: list[str] = []
    install_termination_unwind()
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            if index:
                scene.close()
                scene = build_scene(
                    cfg, level_id=level_id, seed=seed,
                    simplified=args.simplified_graphics,
                )
            detector, gauge, planner, oracle = _parts(args, cfg)
            scene.oracle = oracle
            scene.oracle_heightmap = args.oracle_heightmap
            sink = RunLogSink(log, scene) if log.run_id else None
            if sink is not None:
                sink.begin(seed)
            episode = _run(scene, detector, gauge, planner, seed, speed, sink, args)
            episode_directory = log.directory / str(seed)
            save_snapshots(episode, episode_directory)
            _save_stability(episode_directory, episode.stability_test)
            if sink is not None:
                result = sink.end(episode)
            else:
                result = episode_result(episode, scene)
                log.writer.write(result)
            if args.video:
                _save_video(args.video, episode, index, args.episodes)
            ok += int(episode.success)
            state = episode.states[-1] if episode.states else None
            text = (
                f"semilla {seed}: {episode.n_placed}/{episode.n_objects} · "
                f"{'ÉXITO' if episode.success else 'fallo: ' + str(episode.failure)}"
            )
            if args.show_com and state is not None:
                text += f" · CoG {np_round(state.cog)}"
            lines.append(text)
            _log(args, text)
            # Las líneas del ensayo NO se acumulan aquí: `execute()` ya las emitió con
            # `_log`, y el resumen final sólo imprime `lines[-3:]` —meterlas echaría
            # fuera la línea del episodio y repetiría la de la viga.
            if args.pause and index + 1 < args.episodes:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        _log(args, "interrumpido")
    finally:
        if log.episode_id:
            log.end(_aborted(seed, scene))
        log.close()
        scene.close()

    lines.extend((f"{ok}/{args.episodes} episodios con éxito", f"disco: {log.directory}"))
    if log.path_in_ui:
        lines.append(f"interfaz: {log.path_in_ui}")
    if args.protocol == "json":
        _emit(args, {"kind": "finished", "ok": ok == args.episodes, "lines": lines})
    else:
        for line in lines[-3:]:
            print(line)
    return 0 if ok == args.episodes else 1


def np_round(values) -> list[float]:
    return [round(float(value), 4) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
