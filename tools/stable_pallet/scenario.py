from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .models import Package, Pallet
from .planner import PlannerConfig, PlannerWeights
from .shake import TRANSPORT_AXES, TRANSPORT_LEVELS_G
from .truck import TruckBay


@dataclass(frozen=True, slots=True)
class ShakeConfig:
    """Transport-jolt sweep and narrow-beam check applied after stacking.

    Five peak accelerations, each along X, Y and Z (15 trials), then the loaded pallet
    is seated on a narrow beam, first along X and then along Y. Every trial restores the
    saved post-placement state first. Jolt peaks follow EN 12195-1 and typical
    forklift/road events, starting from a barely perceptible creep.
    """

    levels_g: tuple[float, ...] = TRANSPORT_LEVELS_G
    axes: tuple[str, ...] = TRANSPORT_AXES
    duration: float = 0.50
    settle_seconds: float = 1.5
    hold_seconds: float = 0.5
    rest_seconds: float = 0.35
    max_shift_m: float = 0.030
    beam_width: float = 0.10
    beam_height: float = 0.05
    beam_settle_seconds: float = 2.0
    max_tilt_deg: float = 8.0

    def __post_init__(self) -> None:
        if not self.levels_g:
            raise ValueError("ShakeConfig.levels_g must not be empty")
        if any(level <= 0 for level in self.levels_g):
            raise ValueError("Shake accelerations must be positive")
        unknown = [axis for axis in self.axes if axis not in {"x", "y", "z"}]
        if unknown:
            raise ValueError(f"Unsupported shake axes: {unknown}")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    pallet_origin: tuple[float, float] = (-0.10, -0.50)
    pallet_height: float = 0.12
    infeed_position: tuple[float, float] = (-0.85, 0.0)
    infeed_height: float = 0.62
    settle_seconds: float = 1.25
    position_noise_std: float = 0.0
    friction: float = 0.9
    shake: ShakeConfig = field(default_factory=ShakeConfig)
    # With a bay declared, the cell unloads a loaded trailer instead of taking one
    # singulated carton at a time off the conveyor.
    truck: TruckBay | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    pallet: Pallet
    packages: tuple[Package, ...]
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def _tuple3(value: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("Expected a three-element vector")
    return float(value[0]), float(value[1]), float(value[2])


# Where the demonstrator keeps its own scenarios. It lives under `tools/` in this
# repo, so `scenarios/mixed_boxes.yaml` -- the path every default and every test names
# -- no longer resolves from the repo root. Rather than rewrite forty call sites, a
# relative path that is not where the caller is standing gets one more try here.
_TOOLS_ROOT = Path(__file__).resolve().parent.parent


def resolve_scenario(path: str | Path) -> Path:
    """The scenario file, found from the repo root or from `tools/`, whichever works."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _TOOLS_ROOT / candidate


def load_scenario(path: str | Path) -> Scenario:
    source = resolve_scenario(path)
    data: dict[str, Any]
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)

    pallet = Pallet(**data["pallet"])
    packages = tuple(_package_from_item(item) for item in data["packages"])

    planner_data = data.get("planner", {})
    weights = PlannerWeights(**planner_data.pop("weights", {}))
    planner = PlannerConfig(weights=weights, **planner_data)

    simulation_data = data.get("simulation", {})
    for key in ("pallet_origin", "infeed_position"):
        if key in simulation_data:
            simulation_data[key] = tuple(simulation_data[key])
    truck_data = simulation_data.pop("truck", None)
    simulation_data["truck"] = _truck_from_item(truck_data)
    shake_data = dict(simulation_data.pop("shake", {}))
    shake_data.pop("peak_accel_g", None)
    shake_data.pop("direction", None)
    if "levels_g" in shake_data:
        shake_data["levels_g"] = tuple(float(value) for value in shake_data["levels_g"])
    if "axes" in shake_data:
        shake_data["axes"] = tuple(str(value) for value in shake_data["axes"])
    simulation_data["shake"] = ShakeConfig(**shake_data)
    simulation = SimulationConfig(**simulation_data)
    return Scenario(data.get("name", source.stem), pallet, packages, planner, simulation)


def _truck_from_item(item: dict[str, Any] | bool | None) -> TruckBay | None:
    """`truck:` accepts nothing, a plain switch, or the bay dimensions."""
    if item is None or item is False:
        return None
    if item is True:
        return TruckBay()
    data = dict(item)
    for key in ("origin",):
        if key in data:
            data[key] = tuple(data[key])
    return TruckBay(**data)


def _package_from_item(item: dict[str, Any]) -> Package:
    package_id = str(item["id"])
    size = _tuple3(item["size"])
    volume = size[0] * size[1] * size[2]
    if "density" in item:
        mass = float(item["density"]) * volume
    elif "mass" in item:
        mass = float(item["mass"])
    else:
        raise ValueError(f"Package {package_id} must declare mass or density")
    return Package(
        id=package_id,
        size=size,
        mass=mass,
        com=_tuple3(item.get("com", (0.0, 0.0, 0.0))),
        friction=float(item.get("friction", 0.8)),
    )


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def scenario_as_dict(scenario: Scenario) -> dict[str, Any]:
    packages = []
    for package in scenario.packages:
        item: dict[str, Any] = {
            "id": package.id,
            "size": [round(value, 3) for value in package.size],
            "density": round(package.density, 1),
            "mass": round(package.mass, 3),
            "com": [round(value, 4) for value in package.com],
        }
        if package.friction != 0.8:
            item["friction"] = package.friction
        packages.append(item)
    return {
        "name": scenario.name,
        "pallet": _plain(scenario.pallet),
        "planner": _plain(scenario.planner),
        "simulation": _plain(scenario.simulation),
        "packages": packages,
    }


def dump_scenario(scenario: Scenario, path: str | Path) -> None:
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def _represent_list(dumper: yaml.SafeDumper, data: list[Any]):
        flow = bool(data) and all(isinstance(item, (int, float, str, bool)) for item in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    _Dumper.add_representer(list, _represent_list)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated parcel set. Mass is density × volume (not sampled). CoM is an\n"
        "# offset from the geometric centre, clipped to 25 % of each half-extent,\n"
        "# with typical variation ~10 % and a slight downward bias in height.\n"
    )
    with target.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.dump(scenario_as_dict(scenario), handle, Dumper=_Dumper, sort_keys=False, allow_unicode=True)
