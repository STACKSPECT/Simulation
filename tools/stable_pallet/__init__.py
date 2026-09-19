"""Stable palletizing planner and simulation."""

from .com_estimator import ComEstimate, estimate_com
from .com_probe import ComProbe, PackageMeasurement
from .generator import BoxGenerator, BoxGeneratorConfig, generate_boxes
from .models import Package, Pallet, Placement, StackState
from .pallet_com import PalletCom, PalletComComparison, compare_pallet_com, estimate_pallet_com
from .planner import StablePalletPlanner
from .tare import TareCalibration, calibrate_tare
from .trial import EVAL_BOX_COUNT, EVAL_SEED, build_generated_scenario, run_generated_eval

__all__ = [
    "BoxGenerator",
    "BoxGeneratorConfig",
    "ComEstimate",
    "ComProbe",
    "EVAL_BOX_COUNT",
    "EVAL_SEED",
    "Package",
    "PackageMeasurement",
    "Pallet",
    "PalletCom",
    "PalletComComparison",
    "Placement",
    "StablePalletPlanner",
    "StackState",
    "TareCalibration",
    "build_generated_scenario",
    "calibrate_tare",
    "compare_pallet_com",
    "estimate_com",
    "estimate_pallet_com",
    "generate_boxes",
    "run_generated_eval",
]
__version__ = "0.1.0"
