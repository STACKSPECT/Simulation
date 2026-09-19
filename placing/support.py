from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from placing.heightmap import HeightMap
from placing.types import BoxOrder, FeasibilityConfig


def _pool_bounds(n_dst: int, n_src: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Source cell range [start, stop) covered by each destination cell."""
    k = np.arange(n_dst, dtype=np.float64)
    start = np.clip(np.floor(k * ratio).astype(np.int64), 0, n_src - 1)
    stop = np.clip(np.ceil((k + 1.0) * ratio).astype(np.int64), 1, n_src)
    return start, np.maximum(stop, start + 1)


def resample_height_map(height_map: HeightMap, resolution: float) -> HeightMap:
    """Rebuild a height map at `resolution` using max over overlapping source cells.

    Pooled one axis at a time. Destination blocks overlap by at most a cell, so
    the two passes give the same answer as the per-cell version for a fraction
    of the work: `nx + ny` slices instead of `nx * ny`.
    """
    if abs(height_map.resolution - resolution) < 1e-12:
        return height_map
    nx = max(1, int(round(height_map.length / resolution)))
    ny = max(1, int(round(height_map.width / resolution)))
    ratio = resolution / height_map.resolution
    # An unobserved cell must not contribute a height, so pool -inf in its place
    # and read `observed` back off the result.
    filled = np.where(
        height_map.observed, height_map.heights.astype(np.float64), -np.inf
    )

    i0, i1 = _pool_bounds(nx, height_map.nx, ratio)
    cols = np.empty((height_map.ny, nx), dtype=np.float64)
    for i in range(nx):
        cols[:, i] = filled[:, i0[i] : i1[i]].max(axis=1)

    j0, j1 = _pool_bounds(ny, height_map.ny, ratio)
    pooled = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        pooled[j] = cols[j0[j] : j1[j]].max(axis=0)

    observed = np.isfinite(pooled)
    return HeightMap(
        origin_xy=height_map.origin_xy,
        length=height_map.length,
        width=height_map.width,
        resolution=resolution,
        heights=np.where(observed, pooled, 0.0).astype(np.float32),
        observed=observed,
    )


def cells_for(length: float, resolution: float) -> int:
    return max(1, int(round(length / resolution)))


def clearance_cells(clearance: float, resolution: float) -> int:
    if clearance <= 0:
        return 0
    return max(1, int(math.ceil(clearance / resolution - 1e-12)))


def rotate_xy(
    dx: float | np.ndarray,
    dy: float | np.ndarray,
    yaw_deg: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    yaw = np.deg2rad(yaw_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return c * dx - s * dy, s * dx + c * dy


def com_world_xy(
    x: np.ndarray,
    y: np.ndarray,
    yaw_deg: float,
    com_local: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = rotate_xy(com_local[0], com_local[1], yaw_deg)
    return x + dx, y + dy


def _extent(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last True index along the last axis. All-False rows give (0, -1)."""
    any_true = flags.any(axis=-1)
    first = np.argmax(flags, axis=-1)
    last = flags.shape[-1] - 1 - np.argmax(flags[..., ::-1], axis=-1)
    return np.where(any_true, first, 0), np.where(any_true, last, -1)


def yaw_set(box: BoxOrder, cfg: FeasibilityConfig) -> tuple[float, ...]:
    """Yaws worth enumerating for this box.

    A box whose centre of gravity sits off the geometric centre gets the
    180-degree flips too: turning it round puts the weight on the other side of
    the footprint, which is a different placement, not the same one. For a
    centred CoM the flip is the identical pose, so it is not enumerated.
    """
    yaws = tuple(float(y) for y in cfg.yaws)
    if abs(box.com_local[0]) <= 1e-9 and abs(box.com_local[1]) <= 1e-9:
        return yaws
    return tuple(dict.fromkeys(yaws + tuple((y + 180.0) % 360.0 for y in yaws)))


def footprint_features(
    height_map: HeightMap,
    box: BoxOrder,
    cfg: FeasibilityConfig,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Window features per distinct footprint, so the yaws that share one share it.

    Enumeration and scoring both need these and each sliding-window pass is the
    most expensive thing in the module; computing it once per footprint keeps
    yaw 0/180 (and 90/270) down to a single pass each.
    """
    n_clear = clearance_cells(cfg.clearance, height_map.resolution)
    out: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for yaw in yaw_set(box, cfg):
        foot_l, foot_w = box.footprint(yaw)
        key = (
            cells_for(foot_l, height_map.resolution),
            cells_for(foot_w, height_map.resolution),
        )
        if key not in out:
            out[key] = window_features(
                height_map,
                nx_box=key[0],
                ny_box=key[1],
                n_clear=n_clear,
                support_tol=cfg.support_tol,
            )
    return out


def window_features(
    height_map: HeightMap,
    nx_box: int,
    ny_box: int,
    n_clear: int,
    support_tol: float,
) -> dict[str, np.ndarray]:
    """Vectorized z_base / support / collision / void / unobserved for every window."""
    heights = np.asarray(height_map.heights, dtype=np.float64)
    observed = np.asarray(height_map.observed, dtype=bool)
    ny, nx = heights.shape
    if ny_box > ny or nx_box > nx:
        empty = np.zeros((0, 0), dtype=np.float64)
        empty_b = np.zeros((0, 0), dtype=bool)
        return {
            "z_base": empty,
            "support_ratio": empty,
            "void_volume": empty,
            "collision": empty_b,
            "unobserved": empty_b,
            "inner": empty.reshape(0, 0, ny_box, nx_box),
        }

    pad = n_clear
    h_pad = np.pad(heights, pad, constant_values=-np.inf)
    obs_pad = np.pad(observed, pad, constant_values=True)
    win = (ny_box + 2 * pad, nx_box + 2 * pad)
    outer_h = sliding_window_view(h_pad, win)
    outer_obs = sliding_window_view(obs_pad, win)
    if pad == 0:
        inner_h = outer_h
        inner_obs = outer_obs
    else:
        inner_h = outer_h[:, :, pad : pad + ny_box, pad : pad + nx_box]
        inner_obs = outer_obs[:, :, pad : pad + ny_box, pad : pad + nx_box]

    z_base = inner_h.max(axis=(-2, -1))
    finite_inner = np.isfinite(inner_h)
    z_safe = np.where(finite_inner, inner_h, 0.0)
    z_base_safe = np.where(np.isfinite(z_base), z_base, 0.0)
    supported = finite_inner & ((z_base_safe[..., None, None] - z_safe) <= support_tol)
    support_ratio = supported.reshape(*supported.shape[:2], -1).mean(axis=-1)
    void = np.where(finite_inner, np.clip(z_base_safe[..., None, None] - z_safe, 0.0, None), 0.0)
    void_volume = void.sum(axis=(-2, -1)) * (height_map.resolution**2)

    if pad == 0:
        collision = np.zeros(z_base.shape, dtype=bool)
        ring_unobs = np.zeros(z_base.shape, dtype=bool)
    else:
        outer_finite = np.isfinite(outer_h)
        outer_max = np.where(outer_finite, outer_h, -np.inf).max(axis=(-2, -1))
        collision = outer_max > (z_base_safe + support_tol)
        ring_unobs = ~outer_obs.all(axis=(-2, -1))

    inner_unobs = ~inner_obs.all(axis=(-2, -1))
    unobserved = inner_unobs | ring_unobs

    # Which columns / rows of the footprint touch anything, and the extent they
    # span. That extent is the support polygon a box actually balances on: a box
    # bridging two neighbours rests on nothing in the middle and is still stable,
    # so "is this cell supported" is the wrong question and this is the right one.
    sup_cols = supported.any(axis=-2)
    sup_rows = supported.any(axis=-1)
    x0, x1 = _extent(sup_cols)
    y0, y1 = _extent(sup_rows)
    return {
        "z_base": z_base_safe,
        "support_ratio": support_ratio,
        "void_volume": void_volume,
        "collision": collision,
        "unobserved": unobserved,
        "inner": inner_h,
        "supported": supported,
        "sup_cols": sup_cols,
        "sup_rows": sup_rows,
        "sup_x0": x0,
        "sup_x1": x1,
        "sup_y0": y0,
        "sup_y1": y1,
        "has_support": sup_cols.any(axis=-1),
    }


def flatten_windows(
    height_map: HeightMap,
    box: BoxOrder,
    yaw_deg: float,
    features: dict[str, np.ndarray],
    cfg: FeasibilityConfig,
) -> dict[str, np.ndarray]:
    z_base = features["z_base"]
    ny_pos, nx_pos = z_base.shape
    if ny_pos == 0 or nx_pos == 0:
        return {}
    ny_box, nx_box = features["supported"].shape[-2:]
    jj, ii = np.meshgrid(np.arange(ny_pos, dtype=np.int32), np.arange(nx_pos, dtype=np.int32), indexing="ij")
    res = height_map.resolution
    ox, oy = height_map.origin_xy
    x = ox + (ii.astype(np.float64) + 0.5 * nx_box) * res
    y = oy + (jj.astype(np.float64) + 0.5 * ny_box) * res
    com_x, com_y = com_world_xy(x, y, yaw_deg, box.com_local)
    ci = np.floor((com_x - ox) / res).astype(np.int32)
    cj = np.floor((com_y - oy) / res).astype(np.int32)
    ny, nx = height_map.heights.shape
    in_map = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
    in_foot = (ci >= ii) & (ci < ii + nx_box) & (cj >= jj) & (cj < jj + ny_box)
    ci_clip = np.clip(ci, 0, max(nx - 1, 0))
    cj_clip = np.clip(cj, 0, max(ny - 1, 0))
    com_obs = height_map.observed[cj_clip, ci_clip]
    # The box balances if its centre of gravity falls inside the area it touches,
    # not if the cell directly under the CoG happens to be solid. A box laid
    # across two neighbours carries its weight on both of them and has thin air
    # in the middle; demanding solid ground under the CoG forbids that outright.
    # The extent is the axis-aligned bound of the contact, which is a superset of
    # the true hull: `min_support_ratio` and `overhang` cover what it lets past.
    com_inside = (
        (ci - ii >= features["sup_x0"])
        & (ci - ii <= features["sup_x1"])
        & (cj - jj >= features["sup_y0"])
        & (cj - jj <= features["sup_y1"])
    )
    com_supported = in_map & in_foot & com_obs & features["has_support"] & com_inside

    pad_z = z_base + box.height
    too_tall = pad_z > cfg.max_stack_height + 1e-12
    low_support = features["support_ratio"] < cfg.min_support_ratio
    unobserved = features["unobserved"]
    collision = features["collision"]
    if cfg.allow_unobserved:
        unobs_reject = np.zeros_like(unobserved)
    else:
        unobs_reject = unobserved
    out_of_reach = np.zeros(z_base.shape, dtype=bool)
    if cfg.robot_xy is not None and cfg.reach_max > 0.0:
        dist = np.hypot(x - cfg.robot_xy[0], y - cfg.robot_xy[1])
        # The arm must also hold this xy at transit height, where it carries the
        # box over the tallest stack. That is usually the binding limit, since
        # horizontal reach shrinks faster than the pile grows.
        surface = float(height_map.heights.max()) if height_map.heights.size else 0.0
        transit_z = np.maximum(
            surface + cfg.transit_margin + box.height, pad_z + cfg.transit_min_lift
        )
        limit = np.minimum(cfg.reach_limit(pad_z), cfg.reach_limit(transit_z))
        out_of_reach = (dist > limit) | (dist < cfg.reach_min)

    valid = (
        (~collision)
        & (~unobs_reject)
        & com_supported
        & (~too_tall)
        & (~low_support)
        & (~out_of_reach)
    )
    reason = np.full(z_base.shape, "", dtype=object)
    reason[low_support] = "min_support_ratio"
    reason[too_tall] = "max_stack_height"
    reason[out_of_reach] = "out_of_reach"
    reason[~com_supported] = "com_unsupported"
    reason[unobs_reject] = "unobserved"
    reason[collision] = "collision"
    reason[valid] = ""

    return {
        "x": x.ravel(),
        "y": y.ravel(),
        "yaw": np.full(x.size, float(yaw_deg), dtype=np.float64),
        "z_base": z_base.ravel(),
        "support_ratio": features["support_ratio"].ravel(),
        "void_volume": features["void_volume"].ravel(),
        "collision": collision.ravel(),
        "unobserved": unobserved.ravel(),
        "com_supported": com_supported.ravel(),
        "valid": valid.ravel(),
        "i0": ii.ravel(),
        "j0": jj.ravel(),
        "nx_box": np.full(x.size, nx_box, dtype=np.int32),
        "ny_box": np.full(x.size, ny_box, dtype=np.int32),
        "reject_reason": reason.ravel(),
    }
