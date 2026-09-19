from __future__ import annotations

import numpy as np

from placing.heightmap import HeightMap
from placing.support import (
    cells_for,
    flatten_windows,
    footprint_features,
    yaw_set,
)
from placing.types import (
    BoxOrder,
    CandidateBatch,
    FeasibilityConfig,
    Placement,
    empty_batch,
)


def enumerate_candidates(
    height_map: HeightMap,
    box: BoxOrder,
    cfg: FeasibilityConfig | None = None,
    features: dict[tuple[int, int], dict[str, np.ndarray]] | None = None,
) -> CandidateBatch:
    """Enumerate every discrete axis-aligned pose and mark hard-constraint failures.

    Pass `features` from `support.footprint_features` to share the sliding-window
    pass with the scorer; omitted, it is computed here.
    """
    cfg = cfg or FeasibilityConfig()
    if features is None:
        features = footprint_features(height_map, box, cfg)
    parts: list[dict[str, np.ndarray]] = []
    for yaw in yaw_set(box, cfg):
        foot_l, foot_w = box.footprint(yaw)
        key = (
            cells_for(foot_l, height_map.resolution),
            cells_for(foot_w, height_map.resolution),
        )
        flat = flatten_windows(height_map, box, yaw, features[key], cfg)
        if flat:
            parts.append(flat)
    if not parts:
        return empty_batch()
    stacked = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    return CandidateBatch(**stacked)


def reject_summary(batch: CandidateBatch) -> dict[str, int]:
    """Why nothing fit, as one loggable dict.

    An empty result is a normal outcome, not an exception, and the caller has to
    record *which* constraint did the rejecting or it is debugging blind.
    """
    if not len(batch):
        return {}
    reasons = batch.reject_reason[~batch.valid]
    names, counts = np.unique(reasons, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def pick_best_index(batch: CandidateBatch) -> int | None:
    """Highest support ratio, then lower-left: (z_base, y, x, yaw)."""
    valid_idx = np.flatnonzero(batch.valid)
    if valid_idx.size == 0:
        return None
    order = np.lexsort(
        (
            batch.yaw[valid_idx],
            batch.x[valid_idx],
            batch.y[valid_idx],
            batch.z_base[valid_idx],
            -batch.support_ratio[valid_idx],
        )
    )
    return int(valid_idx[order[0]])


def placement_from_index(batch: CandidateBatch, box: BoxOrder, index: int) -> Placement:
    row = batch.take(index)
    z_base = float(row["z_base"])
    ratio = float(row["support_ratio"])
    foot_l, foot_w = box.footprint(float(row["yaw"]))
    return Placement(
        x=float(row["x"]),
        y=float(row["y"]),
        z=z_base + 0.5 * box.height,
        yaw=float(row["yaw"]),
        z_base=z_base,
        support_ratio=ratio,
        contact_area=ratio * foot_l * foot_w,
        metrics={"support_ratio": ratio},
        score=ratio,
    )


def best_feasible(
    height_map: HeightMap,
    box: BoxOrder,
    cfg: FeasibilityConfig | None = None,
) -> Placement | None:
    batch = enumerate_candidates(height_map, box, cfg)
    index = pick_best_index(batch)
    if index is None:
        return None
    return placement_from_index(batch, box, index)
