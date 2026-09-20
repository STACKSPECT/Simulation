"""Transport jolts and a narrow-beam check applied to the loaded pallet after stacking.

The default protocol follows cargo-securing practice (EN 12195-1 / typical forklift and
road events): five peak accelerations, each applied once along X, Y and Z — 15 trials —
then the same stack is seated on a relatively narrow beam, first along X and then along
Y. Every trial starts from the same saved stack (the pallet and the packages as they were
left after placement). Packages are never welded to the deck; they stay on through
contact and friction only.
"""

from __future__ import annotations

import math

import numpy as np

EURO_PALLET_MASS_KG = 25.0
GRAVITY = 9.81

# Progressive peaks a loaded pallet can see in a warehouse-to-truck journey.
# 0.05 g  slow yard creep
# 0.15 g  gentle vehicle acceleration
# 0.30 g  typical forklift bump / urban braking
# 0.50 g  EN 12195-1 lateral and rearward securing
# 0.80 g  EN 12195-1 forward securing (hard braking)
TRANSPORT_LEVELS_G: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.80)
TRANSPORT_AXES: tuple[str, ...] = ("x", "y", "z")
TRANSPORT_REASONS: dict[float, str] = {
    0.05: "slow yard creep",
    0.15: "gentle vehicle acceleration",
    0.30: "typical forklift bump or urban braking",
    0.50: "EN 12195-1 lateral/rearward cargo securing",
    0.80: "EN 12195-1 forward securing (hard braking)",
}


def axis_unit(axis: str) -> np.ndarray:
    key = axis.lower()
    if key == "x":
        return np.array([1.0, 0.0, 0.0])
    if key == "y":
        return np.array([0.0, 1.0, 0.0])
    if key == "z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Shake axis must be x, y or z, not {axis!r}")


def transport_reason(peak_accel_g: float) -> str:
    return TRANSPORT_REASONS.get(round(peak_accel_g, 4), f"{peak_accel_g:g} g transport jolt")


def sine_jolt_force(elapsed: float, duration: float, peak_force: float, direction: np.ndarray) -> np.ndarray:
    """Push then brake so the pallet starts and ends at rest if nothing slides.

    `a(t) = A sin(2π t / T)` integrates to zero net impulse over one period, which is the
    kinematic signature of a forklift shove or a pothole rather than a sustained haul.
    """
    if duration <= 0.0 or elapsed < 0.0 or elapsed > duration:
        return np.zeros(3)
    return (peak_force * math.sin(2.0 * math.pi * elapsed / duration)) * direction


def package_fell_off(
    relative_center: np.ndarray,
    pallet_width: float,
    pallet_depth: float,
    pallet_height: float,
) -> bool:
    """Whether the package centre has left the deck, in the pallet body frame."""
    return (
        abs(float(relative_center[0])) > pallet_width / 2
        or abs(float(relative_center[1])) > pallet_depth / 2
        or float(relative_center[2]) < pallet_height * 0.5
    )


def pallet_tilt_deg(xmat: np.ndarray, axis: str) -> float:
    """Signed tilt of the pallet deck about a world axis, in degrees."""
    rotation = np.asarray(xmat, dtype=float).reshape(3, 3)
    if axis == "x":
        return math.degrees(math.atan2(rotation[1, 2], rotation[2, 2]))
    if axis == "y":
        return math.degrees(math.atan2(-rotation[0, 2], rotation[2, 2]))
    raise ValueError(f"Beam axis must be x or y, not {axis!r}")


def score_trial(trial: dict, max_shift_m: float) -> float:
    """Score one jolt from 0 to 100.

    100 means the load did not move. 50 is the hold threshold (``max_shift_m``,
    30 mm by default). 0 means a package fell off or slid at least twice that far.
    """
    if max_shift_m <= 0:
        raise ValueError("max_shift_m must be positive")
    packages = trial.get("packages") or []
    if any(item.get("fell_off") for item in packages):
        return 0.0
    if packages:
        worst_m = max(float(item["displacement_mm"]) for item in packages) / 1_000.0
    else:
        worst_m = 0.0 if trial.get("held") else 2.0 * max_shift_m
    return round(max(0.0, min(100.0, 100.0 * (1.0 - worst_m / (2.0 * max_shift_m)))), 1)


def attach_trial_scores(trials: list[dict], max_shift_m: float) -> list[dict]:
    for trial in trials:
        trial["score"] = score_trial(trial, max_shift_m)
    return trials


def summarise_trials(trials: list[dict], axes: tuple[str, ...]) -> dict:
    by_axis: dict[str, dict] = {}
    for axis in axes:
        axis_trials = [trial for trial in trials if trial["axis"] == axis]
        held = [trial for trial in axis_trials if trial["held"]]
        failed = [trial for trial in axis_trials if not trial["held"]]
        axis_summary: dict = {
            "max_held_g": max((trial["peak_accel_g"] for trial in held), default=0.0),
            "first_failure_g": failed[0]["peak_accel_g"] if failed else None,
        }
        axis_scores = [float(trial["score"]) for trial in axis_trials if "score" in trial]
        if axis_scores:
            axis_summary["mean_score"] = round(sum(axis_scores) / len(axis_scores), 1)
            axis_summary["min_score"] = min(axis_scores)
        by_axis[axis] = axis_summary
    summary: dict = {
        "held_all": all(trial["held"] for trial in trials),
        "trial_count": len(trials),
        "by_axis": by_axis,
    }
    scores = [float(trial["score"]) for trial in trials if "score" in trial]
    if scores:
        weights = [float(trial["peak_accel_g"]) for trial in trials if "score" in trial]
        total_weight = sum(weights) or 1.0
        summary["mean_score"] = round(sum(scores) / len(scores), 1)
        summary["min_score"] = min(scores)
        summary["weighted_score"] = round(
            sum(score * weight for score, weight in zip(scores, weights, strict=True)) / total_weight, 1
        )
    return summary


def summarise_beam(trials: list[dict]) -> dict:
    return {
        "held_all": all(trial["held"] for trial in trials),
        "trial_count": len(trials),
        "by_axis": {trial["axis"]: trial["held"] for trial in trials},
    }
