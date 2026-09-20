#!/usr/bin/env python3
"""Mide cuatro cubos, apoya la cara más cercana a su CoM y los lleva al palé."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

from src.com_orientation import (  # noqa: E402
    OrientationExperimentError,
    build_orientation_scene,
    run_orientation_experiment,
)
from src.vision.gauge import WristGauge  # noqa: E402
from src.vision.oracle import OracleDetector  # noqa: E402


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, default=1)
    cli.add_argument("--viewer", action="store_true")
    cli.add_argument(
        "--speed", type=float,
        help="segundos simulados por segundo real; 0 = fast-forward",
    )
    cli.add_argument(
        "--precise-com", action="store_true",
        help="mide el CoM con el barrido de muñeca de cuatro poses",
    )
    cli.add_argument("--simplified-graphics", action="store_true")
    # El mismo protocolo de líneas que `palletize.py`: es lo que el panel sabe leer.
    cli.add_argument("--protocol", choices=("text", "json"), default="text")
    return cli


def _emit(args, payload: dict) -> None:
    if args.protocol == "json":
        print(json.dumps(payload), flush=True)
    elif payload["kind"] == "log":
        print(payload["text"], flush=True)


def _log(args, text: str) -> None:
    _emit(args, {"kind": "log", "text": text})


def _finish(args, ok: bool, lines: list[str]) -> None:
    """Cierra la ejecución igual que el paletizado: una línea por modo, no dos.

    En texto el resumen va a stdout y el fallo a stderr, que es lo que espera quien lo
    corre a mano. En JSON los dos son el mismo `finished`, porque el panel colorea por
    `ok` y no por el descriptor del que vino la línea.
    """
    if args.protocol == "json":
        _emit(args, {"kind": "finished", "ok": ok, "lines": lines})
        return
    for line in lines:
        print(line, file=sys.stdout if ok else sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    speed = args.speed if args.speed is not None else (1.0 if args.viewer else 0.0)
    if speed < 0:
        raise SystemExit("--speed no puede ser negativo")
    scene = build_orientation_scene(
        seed=args.seed, simplified=args.simplified_graphics
    )

    def execute():
        _log(args, "modo: experimento local · CoM de muñeca · sin telemetría")
        return run_orientation_experiment(
            scene,
            OracleDetector(),       # identifica la caja; el CoM NO sale del oráculo
            WristGauge(precise=args.precise_com),
            speed=speed,
            log=lambda line: _log(args, f"  {line}"),
        )

    try:
        if args.viewer:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
                scene.viewer = viewer
                try:
                    result = execute()
                finally:
                    scene.viewer = None
        else:
            result = execute()
    except (KeyboardInterrupt, OrientationExperimentError) as error:
        _finish(args, False, [f"experimento interrumpido: {error}"])
        return 1
    finally:
        scene.close()

    mean_downward = sum(row.downward_cog_m for row in result.records) / len(result.records)
    _finish(args, result.success, [
        f"{len(result.records)}/{len(scene.boxes)} cubos orientados · "
        f"CoM medio {mean_downward * 1000:.1f} mm hacia el apoyo · "
        f"{result.duration_s:.1f} s simulados"
    ])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
