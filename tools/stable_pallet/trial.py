"""End-to-end eval: a loaded trailer, robot palletizing, saved stack, transport scores."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .controls import ViewerControls
from .generator import BoxGeneratorConfig, generate_boxes
from .models import Package
from .scenario import Scenario, ShakeConfig, load_scenario
from .simulator import PalletizingSimulator
from .truck import TruckBay

# Six mixed cartons fill a euro pallet without overflowing it: typically one dense
# layer plus a second, around 0.8 m high. Seed 0 is a closed-loop set that the UR10e
# can pick, place and then shake.
EVAL_BOX_COUNT = 6
EVAL_SEED = 0


def build_generated_scenario(
    count: int = EVAL_BOX_COUNT,
    seed: int = EVAL_SEED,
    *,
    template: str | Path | Scenario = "scenarios/mixed_boxes.yaml",
    name: str | None = None,
    config: BoxGeneratorConfig | None = None,
    source: str = "truck",
) -> Scenario:
    """Random parcels on the template cell, arriving either in a trailer or on the belt."""
    if count <= 0:
        raise ValueError("count must be positive")
    if source not in {"truck", "conveyor"}:
        raise ValueError(f"source must be 'truck' or 'conveyor', not {source!r}")
    base = template if isinstance(template, Scenario) else load_scenario(template)
    packages = generate_boxes(count, seed=seed, config=config)
    bay = (base.simulation.truck or TruckBay()) if source == "truck" else None
    return Scenario(
        name or f"random-{count}-seed-{seed}",
        base.pallet,
        packages,
        base.planner,
        replace(base.simulation, truck=bay),
    )


def _package_record(package: Package) -> dict[str, Any]:
    return {
        "id": package.id,
        "size": list(package.size),
        "mass": package.mass,
        "density": package.density,
        "com": list(package.com),
        "friction": package.friction,
    }


def run_generated_eval(
    *,
    count: int = EVAL_BOX_COUNT,
    seed: int = EVAL_SEED,
    template: str | Path | Scenario = "scenarios/mixed_boxes.yaml",
    viewer: bool = False,
    video_path: str | Path | None = None,
    simplified_graphics: bool = True,
    measure_com: bool = False,
    instant_place: bool = False,
    shake: bool | ShakeConfig = True,
    name: str | None = None,
    controls: ViewerControls | None = None,
    source: str = "truck",
) -> dict[str, Any]:
    """Load ``count`` parcels into the trailer, unload them onto the pallet, then score every jolt."""
    scenario = build_generated_scenario(count, seed, template=template, name=name, source=source)
    simulator = PalletizingSimulator(
        scenario,
        viewer=viewer,
        video_path=video_path,
        seed=seed,
        simplified_graphics=simplified_graphics,
        measure_com=False if instant_place else measure_com,
        controls=controls,
    )
    result = simulator.run(shake=shake, instant_place=instant_place)
    result["generator"] = {
        "count": count,
        "seed": seed,
        "source": source,
        "packages": [_package_record(package) for package in scenario.packages],
    }
    result["placed"] = len(result["state"]["placements"])
    return result
