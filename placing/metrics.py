from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from placing.heightmap import HeightMap
from placing.support import (
    clearance_cells,
    com_world_xy,
    footprint_features,
)
from placing.types import (
    BoxOrder,
    CandidateBatch,
    FeasibilityConfig,
    PalletState,
)

METRIC_NAMES: tuple[str, ...] = (
    "support_ratio",
    "com_margin",
    "lowness",
    "void_fill",
    "levelness",
    "peak_penalty",
    "lateral_proximity",
    "edge_flush",
    "seam_break",
    "overhang",
    "pallet_com",
    "reachability",
    "bridge",
    "gap_waste",
)


@dataclass(frozen=True)
class MetricParams:
    proximity_band: float = 0.02
    levelness_scale: float = 0.2
    max_stack_height: float = 2.0
    support_tol: float = 1e-3
    clearance: float = 0.005
    robot_xy: tuple[float, float] | None = None
    max_reach: float = 0.85
    box_height: float = 0.0
    reach_knee: float = 0.25
    reach_falloff: float = 0.65
    pallet_state: PalletState | None = None
    # Narrowest strip of free deck still worth having. A gap smaller than this
    # holds nothing and is pure waste; a wider one is space, not waste.
    # 0 means "the shorter side of the box being placed".
    useful_gap: float = 0.0


def _integral(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values.astype(np.float64), ((1, 0), (1, 0)))
    return np.cumsum(np.cumsum(padded, axis=0), axis=1)


def _window_sum(sat: np.ndarray, i0: np.ndarray, j0: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return sat[j0 + ny, i0 + nx] - sat[j0, i0 + nx] - sat[j0 + ny, i0] + sat[j0, i0]


def _l1_distance_to(sources: np.ndarray) -> np.ndarray:
    """Batched 4-neighbour (Manhattan) distance to the nearest True cell.

    `sources` is (n, h, w). Manhattan distance is separable, so four sweeps give
    the exact same array as relaxing the whole grid `max(h, w)` times, for about
    a tenth of the work. (The previous name said Chebyshev; the propagation was
    always 4-neighbour, which is this.)
    """
    n, h, w = sources.shape
    dist = np.where(sources, 0, h + w + 2).astype(np.int32)
    for i in range(1, w):
        np.minimum(dist[:, :, i], dist[:, :, i - 1] + 1, out=dist[:, :, i])
    for i in range(w - 2, -1, -1):
        np.minimum(dist[:, :, i], dist[:, :, i + 1] + 1, out=dist[:, :, i])
    for j in range(1, h):
        np.minimum(dist[:, j], dist[:, j - 1] + 1, out=dist[:, j])
    for j in range(h - 2, -1, -1):
        np.minimum(dist[:, j], dist[:, j + 1] + 1, out=dist[:, j])
    return dist


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _spans_a_gap(flags: np.ndarray) -> np.ndarray:
    """True where the footprint touches down at both ends with a gap between.

    `flags` is (n, k): whether each column (or row) of the footprint touches
    anything. Both ends resting on something with air in between is a bridge —
    the box ties two piles into one flat top, which is the point of allowing it.
    """
    n, k = flags.shape
    if k < 3:
        return np.zeros(n, dtype=np.float64)
    end = max(k // 4, 1)
    both_ends = flags[:, :end].any(axis=1) & flags[:, -end:].any(axis=1)
    middle = ~flags[:, end:-end].all(axis=1)
    return both_ends & middle


def _bridge_scores(sup_cols: np.ndarray, sup_rows: np.ndarray) -> np.ndarray:
    """1 where the box spans a gap along either axis, 0 otherwise."""
    return (_spans_a_gap(sup_cols) | _spans_a_gap(sup_rows)).astype(np.float64)


def _gap_waste(
    height_map: HeightMap,
    i0: np.ndarray,
    j0: np.ndarray,
    nx_box: int,
    ny_box: int,
    z_base: np.ndarray,
    n_clear: int,
    n_useful: int,
    support_tol: float,
) -> np.ndarray:
    """1 when the box leaves no unusable strip beside it, 0 when it leaves four.

    For each side, the free deck between the box and the next thing taller than
    it, or the pallet edge. Flush wastes nothing. A gap wider than `n_useful`
    wastes nothing either — that is still room for another box, so sitting in
    the middle is fine as long as what is left on both sides stays usable. What
    is punished is the strip in between: too narrow to fill, too wide to ignore,
    and it is how a pallet quietly loses a fifth of its deck one placement at a
    time.

    The mandatory clearance is discounted against a neighbour, because that air
    has to be there, but not against the pallet edge, where a box can sit right
    up against the rim.
    """
    heights = np.asarray(height_map.heights, dtype=np.float64)
    ny, nx = heights.shape
    # Tallest cell over the footprint's own span, per column and per row, so a
    # side only has to look along one axis.
    col_max = sliding_window_view(heights, ny_box, axis=0).max(axis=-1).T  # (nx, ny_pos)
    row_max = sliding_window_view(heights, nx_box, axis=1).max(axis=-1)    # (ny, nx_pos)
    step = np.arange(n_useful + n_clear + 1, dtype=np.int64)
    limit = z_base[:, None] + support_tol
    rows = np.arange(i0.size)

    def _waste(index: np.ndarray, along: np.ndarray, at: np.ndarray, size: int) -> np.ndarray:
        outside = (index < 0) | (index >= size)
        taller = along[np.clip(index, 0, size - 1), at[:, None]] > limit
        blocked = outside | taller
        hit = blocked.any(axis=1)
        first = blocked.argmax(axis=1)
        # The pallet edge needs no clearance — a box can sit right against it.
        # A neighbour does, and that mandatory air is not waste.
        allowance = np.where(outside[rows, first], 0, n_clear)
        gap = np.maximum(first - allowance, 0)
        return np.where(hit & (gap < n_useful), gap, 0)

    wasted = np.stack(
        [
            _waste(i0[:, None] - 1 - step, col_max, j0, nx),
            _waste(i0[:, None] + nx_box + step, col_max, j0, nx),
            _waste(j0[:, None] - 1 - step, row_max, i0, ny),
            _waste(j0[:, None] + ny_box + step, row_max, i0, ny),
        ],
        axis=1,
    )
    return _clip01(1.0 - wasted.mean(axis=1) / n_useful)


def _pallet_com_scores(
    batch: CandidateBatch,
    box: BoxOrder,
    params: MetricParams,
) -> np.ndarray:
    """[0, 1] closeness of the resulting stack CoM to the pallet centre.

    Neutral (all 1.0) until `params.pallet_state` carries a pallet centre.
    Uses BoxOrder.mass / com_local, never simulation state.
    """
    n = len(batch)
    state = params.pallet_state
    if state is None or state.pallet_center_xy is None or state.pallet_half_diag <= 1e-12:
        return np.ones(n, dtype=np.float64)
    cx, cy = state.pallet_center_xy
    scores = np.ones(n, dtype=np.float64)
    mass = max(float(box.mass), 0.0)
    for yaw in np.unique(batch.yaw):
        sel = np.flatnonzero(batch.yaw == yaw)
        if sel.size == 0:
            continue
        bx, by = com_world_xy(batch.x[sel], batch.y[sel], float(yaw), box.com_local)
        if state.mass <= 1e-12:
            nx, ny = bx, by
        else:
            total = state.mass + mass
            nx = (state.mass * state.com_xy[0] + mass * bx) / total
            ny = (state.mass * state.com_xy[1] + mass * by) / total
        dist = np.hypot(nx - cx, ny - cy)
        scores[sel] = _clip01(1.0 - dist / state.pallet_half_diag)
    return scores


def compute_metrics(
    height_map: HeightMap,
    box: BoxOrder,
    batch: CandidateBatch,
    params: MetricParams | None = None,
    cfg: FeasibilityConfig | None = None,
    features: dict[tuple[int, int], dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Return a [0, 1] array per metric, aligned with `batch`.

    The window-derived terms are only filled for candidates `batch` marked valid;
    the rejected ones score `-inf` downstream and their metrics are never read.
    That is typically 98 % of the batch, and by far the most expensive part.

    Pass `features` from `support.footprint_features` to share the sliding-window
    pass with `enumerate_candidates`; omitted, it is computed here.
    """
    params = params or MetricParams()
    cfg = cfg or FeasibilityConfig(
        clearance=params.clearance,
        support_tol=params.support_tol,
        max_stack_height=params.max_stack_height,
    )
    n = len(batch)
    out = {name: np.zeros(n, dtype=np.float64) for name in METRIC_NAMES}
    if n == 0:
        out["pallet_com"] = np.ones(0, dtype=np.float64)
        out["reachability"] = np.ones(0, dtype=np.float64)
        return out

    out["support_ratio"] = np.clip(batch.support_ratio.astype(np.float64), 0.0, 1.0)
    out["pallet_com"] = _pallet_com_scores(batch, box, params)
    if params.robot_xy is None:
        out["reachability"] = np.ones(n, dtype=np.float64)
    else:
        rx, ry = params.robot_xy
        dist = np.hypot(batch.x - rx, batch.y - ry)
        pad_z = batch.z_base + max(params.box_height, box.height)
        above = np.maximum(pad_z - params.reach_knee, 0.0)
        limit = np.maximum(params.max_reach - params.reach_falloff * above, 1e-6)
        out["reachability"] = _clip01((limit - dist) / limit)

    max_h = max(params.max_stack_height, 1e-6)
    out["lowness"] = _clip01(1.0 - batch.z_base / max_h)

    current_peak = float(height_map.heights.max()) if height_map.heights.size else 0.0
    new_peak = np.maximum(current_peak, batch.z_base + box.height)
    out["peak_penalty"] = _clip01(1.0 - np.maximum(0.0, new_peak - current_peak) / max(box.height, 1e-6))

    area = np.maximum(batch.nx_box * batch.ny_box * (height_map.resolution**2), 1e-12)
    denom = area * np.maximum(batch.z_base, height_map.resolution)
    void_fill = np.ones(n, dtype=np.float64)
    positive = batch.z_base > params.support_tol
    void_fill[positive] = 1.0 - np.clip(batch.void_volume[positive] / denom[positive], 0.0, 1.0)
    out["void_fill"] = _clip01(void_fill)

    sat_h = _integral(height_map.heights)
    sat_sq = _integral(np.square(height_map.heights.astype(np.float64)))
    n_cells = float(height_map.heights.size)
    s1 = float(height_map.heights.sum())
    s2 = float(np.square(height_map.heights.astype(np.float64)).sum())
    scale = max(params.levelness_scale, 1e-6)
    n_clear = clearance_cells(cfg.clearance, height_map.resolution)
    n_prox = clearance_cells(params.proximity_band, height_map.resolution)
    if features is None:
        features = footprint_features(height_map, box, cfg)

    for yaw in np.unique(batch.yaw):
        sel = np.flatnonzero((batch.yaw == yaw) & batch.valid)
        if sel.size == 0:
            continue
        nx_box = int(batch.nx_box[sel[0]])
        ny_box = int(batch.ny_box[sel[0]])
        i0 = batch.i0[sel]
        j0 = batch.j0[sel]
        z_base = batch.z_base[sel]
        old_sum = _window_sum(sat_h, i0, j0, nx_box, ny_box)
        old_sq = _window_sum(sat_sq, i0, j0, nx_box, ny_box)
        z_top = z_base + box.height
        win_n = float(nx_box * ny_box)
        new_sum = s1 - old_sum + win_n * z_top
        new_sq = s2 - old_sq + win_n * (z_top**2)
        var = np.maximum(new_sq / n_cells - (new_sum / n_cells) ** 2, 0.0)
        out["levelness"][sel] = _clip01(1.0 / (1.0 + np.sqrt(var) / scale))

        feat = features[(nx_box, ny_box)]
        supported = feat["supported"][j0, i0]
        foot_l = nx_box * height_map.resolution
        foot_w = ny_box * height_map.resolution
        half_diag = 0.5 * float(np.hypot(foot_l, foot_w))
        com_x, com_y = com_world_xy(batch.x[sel], batch.y[sel], float(yaw), box.com_local)
        ox, oy = height_map.origin_xy
        res = height_map.resolution

        # Stability margin: how far the CoG sits from the edge of the area the
        # box actually rests on. For a box bridging two neighbours that area
        # spans both of them, so a centred CoG scores well even though the cell
        # under it is thin air — which is the whole point of a bridge.
        ci = np.floor((com_x - ox) / res).astype(np.int64) - i0
        cj = np.floor((com_y - oy) / res).astype(np.int64) - j0
        margin_cells = np.minimum.reduce(
            [
                ci - feat["sup_x0"][j0, i0],
                feat["sup_x1"][j0, i0] - ci,
                cj - feat["sup_y0"][j0, i0],
                feat["sup_y1"][j0, i0] - cj,
            ]
        ).astype(np.float64)
        out["com_margin"][sel] = _clip01((margin_cells * res) / max(half_diag, 1e-6))
        out["com_margin"][sel] = np.where(batch.com_supported[sel], out["com_margin"][sel], 0.0)

        out["bridge"][sel] = _bridge_scores(
            feat["sup_cols"][j0, i0], feat["sup_rows"][j0, i0]
        )

        unsupported = ~supported
        if unsupported.any():
            padded = np.pad(supported, ((0, 0), (1, 1), (1, 1)), constant_values=False)
            dist_to_support = _l1_distance_to(padded)[:, 1:-1, 1:-1].astype(np.float64)
            dist_to_support[~unsupported] = 0.0
            max_cant = dist_to_support.reshape(sel.size, -1).max(axis=1)
        else:
            max_cant = np.zeros(sel.size, dtype=np.float64)
        out["overhang"][sel] = _clip01(1.0 - (max_cant * res) / max(max(foot_l, foot_w), 1e-6))

        n_useful = max(
            1,
            int(round((params.useful_gap or min(foot_l, foot_w)) / res)),
        )
        out["gap_waste"][sel] = _gap_waste(
            height_map, i0, j0, nx_box, ny_box, z_base, n_clear, n_useful, cfg.support_tol
        )

        lat, flush, seam = _side_metrics(
            height_map,
            i0,
            j0,
            nx_box,
            ny_box,
            z_base,
            n_clear,
            n_prox,
            cfg.support_tol,
        )
        out["lateral_proximity"][sel] = lat
        out["edge_flush"][sel] = flush
        out["seam_break"][sel] = seam

    for name in METRIC_NAMES:
        out[name] = _clip01(out[name])
    return out


def _side_metrics(
    height_map: HeightMap,
    i0: np.ndarray,
    j0: np.ndarray,
    nx_box: int,
    ny_box: int,
    z_base: np.ndarray,
    n_clear: int,
    n_prox: int,
    support_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights = np.asarray(height_map.heights, dtype=np.float64)
    ny, nx = heights.shape
    pad = n_clear + max(n_prox, 1)
    h_pad = np.pad(heights, pad, constant_values=-np.inf)
    win = (ny_box + 2 * pad, nx_box + 2 * pad)
    outer = sliding_window_view(h_pad, win)
    windows = outer[j0, i0]
    n = i0.size
    footprint = windows[:, pad : pad + ny_box, pad : pad + nx_box]
    z = z_base[:, None, None]

    def _strip(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        block = windows[:, y0:y1, x0:x1]
        if block.size == 0:
            return np.full(n, -np.inf)
        finite = np.isfinite(block)
        filled = np.where(finite, block, -np.inf)
        return filled.reshape(n, -1).max(axis=1)

    left_prox = _strip(pad, pad + ny_box, pad - n_clear - n_prox, pad - n_clear)
    right_prox = _strip(pad, pad + ny_box, pad + nx_box + n_clear, pad + nx_box + n_clear + n_prox)
    bottom_prox = _strip(pad - n_clear - n_prox, pad - n_clear, pad, pad + nx_box)
    top_prox = _strip(pad + ny_box + n_clear, pad + ny_box + n_clear + n_prox, pad, pad + nx_box)

    left_edge = i0 <= max(n_clear + n_prox - 1, 0)
    right_edge = (i0 + nx_box) >= (nx - max(n_clear + n_prox - 1, 0))
    bottom_edge = j0 <= max(n_clear + n_prox - 1, 0)
    top_edge = (j0 + ny_box) >= (ny - max(n_clear + n_prox - 1, 0))

    tall = z_base + support_tol
    lateral = np.stack(
        [
            (left_prox > tall) | left_edge,
            (right_prox > tall) | right_edge,
            (bottom_prox > tall) | bottom_edge,
            (top_prox > tall) | top_edge,
        ],
        axis=1,
    ).mean(axis=1)

    left_adj = windows[:, pad : pad + ny_box, pad - 1]
    right_adj = windows[:, pad : pad + ny_box, pad + nx_box]
    bottom_adj = windows[:, pad - 1, pad : pad + nx_box]
    top_adj = windows[:, pad + ny_box, pad : pad + nx_box]
    left_in = footprint[:, :, 0]
    right_in = footprint[:, :, -1]
    bottom_in = footprint[:, 0, :]
    top_in = footprint[:, -1, :]

    def _drop(inside: np.ndarray, outside: np.ndarray) -> np.ndarray:
        outside_h = np.where(np.isfinite(outside), outside, -np.inf)
        return ((inside - outside_h) > support_tol).mean(axis=1)

    # A side lying on the pallet rim is not a seam, it is the end of the pallet,
    # so it is left out of the average rather than counted as a seam running the
    # full length. Counting it pushed boxes off the rim to "break" it, which is
    # the opposite of what a first layer should do.
    seams = np.stack(
        [
            _drop(left_in, left_adj),
            _drop(right_in, right_adj),
            _drop(bottom_in, bottom_adj),
            _drop(top_in, top_adj),
        ],
        axis=1,
    )
    on_rim = np.stack(
        [i0 == 0, i0 + nx_box == nx, j0 == 0, j0 + ny_box == ny], axis=1
    )
    inner = ~on_rim
    n_inner = inner.sum(axis=1)
    seam_align = np.where(
        n_inner > 0, (seams * inner).sum(axis=1) / np.maximum(n_inner, 1), 0.0
    )
    seam_break = _clip01(1.0 - seam_align)

    def _corner(inside: np.ndarray, out_a: np.ndarray, out_b: np.ndarray) -> np.ndarray:
        da = np.abs(inside - np.where(np.isfinite(out_a), out_a, inside)) > support_tol
        db = np.abs(inside - np.where(np.isfinite(out_b), out_b, inside)) > support_tol
        return (da | db).astype(np.float64)

    c00 = _corner(footprint[:, 0, 0], bottom_adj[:, 0], left_adj[:, 0])
    c01 = _corner(footprint[:, 0, -1], bottom_adj[:, -1], right_adj[:, 0])
    c10 = _corner(footprint[:, -1, 0], top_adj[:, 0], left_adj[:, -1])
    c11 = _corner(footprint[:, -1, -1], top_adj[:, -1], right_adj[:, -1])
    flush = _clip01(0.25 * (c00 + c01 + c10 + c11))
    return _clip01(lateral), flush, seam_break
