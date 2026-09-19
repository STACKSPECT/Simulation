#!/usr/bin/env python3
"""Ejecuta la celda autónoma con una de las tres fuentes y nueve niveles."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402
from theker_telemetry import EpisodeResult, RunLog  # noqa: E402

from src.cell.scene import build_scene, levels, load_configs  # noqa: E402
from src.episode import run_episode  # noqa: E402
from src.planner.heuristic import ScorePlanner  # noqa: E402
from src.planner.naive import BeamPlanner, GridPlanner  # noqa: E402
from src.telemetry import RunLogSink, episode_result, run_config, save_snapshots  # noqa: E402
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
                     help="0 = fast-forward; por defecto 1 con visor y 0 sin visor")
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
    cli.add_argument(
        "--precise-com",
        action="store_true",
        help="inclina la muñeca en varias poses en vez de una lectura a plomo",
    )
    cli.add_argument("--naive-planner", action="store_true")
    cli.add_argument("--score-planner", action="store_true",
                     help="usa heuristic.ScorePlanner cuando su equipo lo implemente")
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
    if args.naive_planner:
        planner = GridPlanner(cfg)
    elif args.score_planner:
        planner = ScorePlanner(cfg)
    else:
        planner = BeamPlanner(cfg)
    oracle = oracle_for(args.oracle_vision, args.oracle_gauge, args.naive_planner)
    return detector, gauge, planner, oracle


def oracle_for(oracle_vision: bool, oracle_gauge: bool, naive_planner: bool) -> bool:
    """El run es oráculo si cualquiera de sus tres piezas es un stub."""
    return any((oracle_vision, oracle_gauge, naive_planner))


def _run(scene, detector, gauge, planner, seed: int, speed: float, sink, args):
    if not args.viewer:
        return run_episode(
            scene, detector, gauge, planner, seed=seed, speed=speed,
            sink=sink, verbose=args.protocol == "text",
        )
    import mujoco.viewer

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        scene.viewer = viewer
        try:
            return run_episode(
                scene, detector, gauge, planner, seed=seed, speed=speed,
                sink=sink, verbose=args.protocol == "text",
            )
        finally:
            scene.viewer = None


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
        task="palletizing",
        metrics={"aborted": True},
    )


def _save_video(path: Path, episode, index: int, total: int) -> None:
    frames = [shot.image for shot in episode.snapshots if shot.view == "top"]
    if not frames:
        return
    destination = path if total == 1 else path.with_stem(f"{path.stem}-{index:03d}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(destination, frames, fps=1, codec="libx264")


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
    log = RunLog(
        REPO,
        task="palletizing",
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
    _log(args, f"modo: {'oráculo' if oracle else 'sin oráculo'} · "
               f"{first_scene.level.name} · {args.episodes} episodio(s)")

    ok = 0
    seed = args.seed
    scene = first_scene
    lines: list[str] = []
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
            sink = RunLogSink(log, scene) if log.run_id else None
            if sink is not None:
                sink.begin(seed)
            episode = _run(scene, detector, gauge, planner, seed, speed, sink, args)
            save_snapshots(episode, log.directory / str(seed))
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
