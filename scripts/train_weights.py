#!/usr/bin/env python3
"""Entrena los 14 pesos de colocación con MuJoCo, sin ejecutar el robot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.cell.scene import levels, load_configs  # noqa: E402
from src.training import train_weights  # noqa: E402


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--levels", type=int, nargs="+", default=[16],
        help="niveles con los que entrenar (por defecto: 16)",
    )
    cli.add_argument(
        "--seeds", type=int, nargs="+", default=[1],
        help="semillas evaluadas por candidato (por defecto: 1)",
    )
    cli.add_argument("--iterations", type=int, default=3)
    cli.add_argument("--population", type=int, default=4)
    cli.add_argument("--elite-fraction", type=float, default=0.5)
    cli.add_argument(
        "--sigma", type=float, default=0.35,
        help="dispersión inicial de los pesos en espacio logarítmico",
    )
    cli.add_argument("--training-seed", type=int, default=2026)
    cli.add_argument(
        "--output", type=Path,
        help="directorio nuevo para manifest.json, progress.jsonl y resultados",
    )
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    cfg = load_configs(REPO)
    available = levels(cfg)
    unknown = sorted(set(args.levels) - set(available))
    if unknown:
        cli.error(f"niveles desconocidos: {unknown}; disponibles: {sorted(available)}")
    if any(seed < 0 for seed in args.seeds):
        cli.error("las semillas tienen que ser no negativas")

    scenarios = len(args.levels) * len(args.seeds)
    evaluations = args.iterations * args.population * scenarios
    print(
        "entrenamiento sin robot · "
        f"niveles {args.levels} · semillas {args.seeds} · "
        f"{evaluations} evaluaciones físicas"
    )
    result = train_weights(
        levels=args.levels,
        seeds=args.seeds,
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        sigma=args.sigma,
        training_seed=args.training_seed,
        cfg=cfg,
        output_directory=args.output,
    )
    print(f"pesos: {result.output_directory / 'best_weights.json'}")
    print(f"progreso: {result.output_directory / 'progress.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
