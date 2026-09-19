from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .models import Package

# Millimetre-level dimensions; 0.1 mm for the centre of mass; 0.1 kg/m³ for density.
_SIZE_DECIMALS = 3
_COM_DECIMALS = 4
_DENSITY_DECIMALS = 1
_SAMPLE_ATTEMPTS = 10_000


@dataclass(frozen=True, slots=True)
class BoxGeneratorConfig:
    """Envelope for a random cardboard-parcel generator.

    Defaults match packed e-commerce cartons a UR10e + OnRobot VGP20 can lift:

    - size 18–55 × 16–40 × 8–30 cm, volume 4–50 L, longest side ≤ 4× shortest
    - density 90–280 kg/m³ (light apparel/void fill up to dense books)
    - mass 0.5–8.5 kg, never sampled: always density × volume; 8.5 kg is the
      box payload cap (12.5 kg UR10e − 2.55 kg VGP20, with margin)
    - CoM offset from the geometric centre, clipped to 25 % of each half-extent,
      sampled with σ ≈ 10 % of that half-extent and a slight downward bias in
      height (~4 % of the vertical half-extent) because packed contents settle
    """

    min_length: float = 0.18
    max_length: float = 0.55
    min_width: float = 0.16
    max_width: float = 0.40
    min_height: float = 0.08
    max_height: float = 0.30
    min_volume: float = 0.004
    max_volume: float = 0.050
    # Packed e-commerce cartons: light apparel/void fill up to dense books. Solid
    # water is 1000 kg/m³; empty cardboard is ~50. This band stays in between.
    min_density: float = 90.0
    max_density: float = 280.0
    min_mass: float = 0.5
    max_mass: float = 8.5
    max_aspect_ratio: float = 4.0
    # CoM offset is a fraction of each half-extent. 0.25 keeps it in the inner
    # half of the box, so a cube cannot place its CoM on a face, let alone a corner.
    com_sigma_ratio: float = 0.10
    com_max_ratio: float = 0.25
    # Packed contents settle slightly below the geometric centre.
    com_z_bias_ratio: float = 0.04
    friction: float = 0.8

    def __post_init__(self) -> None:
        pairs = (
            ("length", self.min_length, self.max_length),
            ("width", self.min_width, self.max_width),
            ("height", self.min_height, self.max_height),
            ("volume", self.min_volume, self.max_volume),
            ("density", self.min_density, self.max_density),
            ("mass", self.min_mass, self.max_mass),
        )
        for name, low, high in pairs:
            if low <= 0 or high <= 0:
                raise ValueError(f"{name} limits must be positive")
            if low > high:
                raise ValueError(f"{name} minimum cannot exceed its maximum")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be at least 1")
        if not 0 < self.com_sigma_ratio <= self.com_max_ratio <= 0.5:
            raise ValueError("CoM ratios must satisfy 0 < sigma <= max <= 0.5")
        if not 0 <= self.com_z_bias_ratio < self.com_max_ratio:
            raise ValueError("com_z_bias_ratio must lie inside the CoM clip window")
        if self.friction <= 0:
            raise ValueError("friction must be positive")
        smallest = self.min_length * self.min_width * self.min_height
        largest = self.max_length * self.max_width * self.max_height
        if self.max_volume < smallest or self.min_volume > largest:
            raise ValueError("volume limits are outside the reachable dimension box")


class BoxGenerator:
    def __init__(self, config: BoxGeneratorConfig | None = None, seed: int | None = None) -> None:
        self.config = config or BoxGeneratorConfig()
        self._rng = Random(seed)

    def generate(self, package_id: str) -> Package:
        cfg = self.config
        for _ in range(_SAMPLE_ATTEMPTS):
            size = (
                round(self._rng.uniform(cfg.min_length, cfg.max_length), _SIZE_DECIMALS),
                round(self._rng.uniform(cfg.min_width, cfg.max_width), _SIZE_DECIMALS),
                round(self._rng.uniform(cfg.min_height, cfg.max_height), _SIZE_DECIMALS),
            )
            if min(size) <= 0:
                continue
            volume = size[0] * size[1] * size[2]
            if not cfg.min_volume <= volume <= cfg.max_volume:
                continue
            longest, shortest = max(size), min(size)
            if longest / shortest > cfg.max_aspect_ratio:
                continue
            density_lo = max(cfg.min_density, cfg.min_mass / volume)
            density_hi = min(cfg.max_density, cfg.max_mass / volume)
            if density_lo > density_hi:
                continue
            density = round(self._rng.uniform(density_lo, density_hi), _DENSITY_DECIMALS)
            mass = density * volume
            if not cfg.min_mass <= mass <= cfg.max_mass:
                continue
            com = tuple(round(offset, _COM_DECIMALS) for offset in self._sample_com(size))
            return Package(package_id, size, mass, com, cfg.friction)
        raise RuntimeError("Could not sample a feasible box; check generator limits")

    def generate_many(self, count: int, id_prefix: str = "BOX") -> tuple[Package, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        width = max(2, len(str(count)))
        return tuple(self.generate(f"{id_prefix}-{index:0{width}d}") for index in range(1, count + 1))

    def _sample_com(self, size: tuple[float, float, float]) -> tuple[float, float, float]:
        cfg = self.config
        offsets: list[float] = []
        for axis, extent in enumerate(size):
            half = extent / 2
            mean = -cfg.com_z_bias_ratio * half if axis == 2 else 0.0
            offset = self._rng.gauss(mean, cfg.com_sigma_ratio * half)
            limit = cfg.com_max_ratio * half
            offsets.append(max(-limit, min(limit, offset)))
        return offsets[0], offsets[1], offsets[2]


def generate_boxes(
    count: int,
    seed: int = 7,
    config: BoxGeneratorConfig | None = None,
    id_prefix: str = "BOX",
) -> tuple[Package, ...]:
    return BoxGenerator(config, seed).generate_many(count, id_prefix)
