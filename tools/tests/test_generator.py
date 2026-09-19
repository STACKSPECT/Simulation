from math import hypot
from pathlib import Path

import pytest

from stable_pallet.cli import build_parser
from stable_pallet.generator import BoxGenerator, BoxGeneratorConfig, generate_boxes
from stable_pallet.models import Package
from stable_pallet.scenario import load_scenario
from stable_pallet.simulator import PalletizingSimulator


def test_package_volume_and_density() -> None:
    package = Package("cube", (0.4, 0.5, 0.2), 8.0)
    assert package.volume == pytest.approx(0.04)
    assert package.density == pytest.approx(200.0)


def test_generated_boxes_stay_inside_limits() -> None:
    config = BoxGeneratorConfig()
    packages = BoxGenerator(config, seed=7).generate_many(40)
    assert len(packages) == 40
    assert [package.id for package in packages[:2]] == ["BOX-01", "BOX-02"]

    for package in packages:
        length, width, height = package.size
        assert config.min_length <= length <= config.max_length
        assert config.min_width <= width <= config.max_width
        assert config.min_height <= height <= config.max_height
        assert config.min_volume <= package.volume <= config.max_volume
        assert config.min_density <= package.density <= config.max_density
        assert config.min_mass <= package.mass <= config.max_mass
        assert package.mass == pytest.approx(package.density * package.volume)
        assert max(package.size) / min(package.size) <= config.max_aspect_ratio
        assert package.mass + PalletizingSimulator.tool_mass_kg <= PalletizingSimulator.ur10e_payload_kg


def test_mass_is_density_times_volume_so_larger_boxes_tend_heavier() -> None:
    packages = generate_boxes(80, seed=11)
    by_volume = sorted(packages, key=lambda item: item.volume)
    light_half = by_volume[: len(by_volume) // 2]
    heavy_half = by_volume[len(by_volume) // 2 :]
    assert sum(item.mass for item in heavy_half) / len(heavy_half) > sum(item.mass for item in light_half) / len(
        light_half
    )


def test_centre_of_mass_stays_in_the_inner_core() -> None:
    config = BoxGeneratorConfig()
    packages = BoxGenerator(config, seed=23).generate_many(60)
    for package in packages:
        normalized = []
        for offset, extent in zip(package.com, package.size, strict=True):
            half = extent / 2
            ratio = abs(offset) / half
            assert ratio <= config.com_max_ratio + 1e-9
            normalized.append(ratio)
        # A corner of the box would sit at normalised radius 1. The allowed
        # inner cube peaks at 0.25 * sqrt(3) ≈ 0.43.
        assert hypot(*normalized) <= config.com_max_ratio * (3**0.5) + 1e-9

    mean_x = sum(package.com[0] for package in packages) / len(packages)
    mean_y = sum(package.com[1] for package in packages) / len(packages)
    mean_z = sum(package.com[2] for package in packages) / len(packages)
    assert abs(mean_x) < 0.01
    assert abs(mean_y) < 0.01
    assert mean_z < 0.0


def test_a_cube_cannot_place_its_com_in_a_corner() -> None:
    config = BoxGeneratorConfig(
        min_length=0.30,
        max_length=0.32,
        min_width=0.30,
        max_width=0.32,
        min_height=0.30,
        max_height=0.32,
        min_volume=0.027,
        max_volume=0.033,
        max_aspect_ratio=1.15,
    )
    packages = BoxGenerator(config, seed=3).generate_many(30)
    for package in packages:
        half = tuple(extent / 2 for extent in package.size)
        corner = hypot(*half)
        distance = hypot(*package.com)
        assert distance / corner < 0.5


def test_generator_is_deterministic_for_a_seed() -> None:
    first = generate_boxes(5, seed=99)
    second = generate_boxes(5, seed=99)
    third = generate_boxes(5, seed=100)
    assert first == second
    assert first != third


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        BoxGeneratorConfig(min_length=0.5, max_length=0.2)
    with pytest.raises(ValueError, match="CoM"):
        BoxGeneratorConfig(com_max_ratio=0.8)
    with pytest.raises(ValueError, match="volume limits"):
        BoxGeneratorConfig(min_volume=0.2, max_volume=0.3)


def test_scenario_can_be_declared_by_density(tmp_path: Path) -> None:
    source = tmp_path / "density.yaml"
    source.write_text(
        """
name: density-only
pallet: {width: 1.2, depth: 0.8}
packages:
  - id: DENSE
    size: [0.40, 0.20, 0.10]
    density: 200.0
    com: [0.01, 0.0, 0.0]
""",
        encoding="utf-8",
    )
    package = load_scenario(source).packages[0]
    assert package.mass == pytest.approx(1.6)
    assert package.density == pytest.approx(200.0)


def test_generated_scenario_round_trips(tmp_path: Path) -> None:
    parser = build_parser()
    output = tmp_path / "random.yaml"
    args = parser.parse_args(
        [
            "generate",
            "--count",
            "6",
            "--seed",
            "7",
            "--output",
            str(output),
            "--name",
            "round-trip",
        ]
    )
    assert args.handler(args) == 0
    loaded = load_scenario(output)
    assert loaded.name == "round-trip"
    assert len(loaded.packages) == 6
    original = BoxGenerator(seed=7).generate_many(6)
    for generated, restored in zip(original, loaded.packages, strict=True):
        assert restored.id == generated.id
        assert restored.size == generated.size
        assert restored.density == pytest.approx(generated.density, abs=0.05)
        assert restored.mass == pytest.approx(generated.mass, rel=1e-3)
        assert restored.com == generated.com
