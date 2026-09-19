"""
La superficie del montón, vista desde las cámaras: profundidad -> puntos -> rejilla.

Portado de `pallet_perception/perception/{unproject,rasterize,fuse}.py`. No se reescribe
"mejor": son las cuentas que ya estaban medidas contra la verdad de la escena.

**numpy y nada más.** Ni mujoco, ni lector de YAML, ni nada que sepa que existe un
simulador. Lo que entra son fotogramas de profundidad ya posados (`DepthFrame`) y lo que
sale es una rejilla de alturas. Esa frontera es lo que permite probar la fusión con un
fotograma sintético, en milisegundos y sin arrancar MuJoCo: ver `tests/test_pallet.py`.
El lado que sí toca el simulador vive al lado, en `src/vision/depth.py`.

Tres cosas que cuestan un rato descubrir y aquí van escritas:

  - **La profundidad de MuJoCo es Z planar de cámara, no longitud de rayo.** Tratarla
    como distancia curva el palé hacia los bordes del encuadre.
  - **La cámara mira por su −Z**, con +X a la derecha y +Y arriba; la fila 0 de la imagen
    es la de arriba, así que al pasar de píxel a cámara hay que invertir Y.
  - **Se tiran los costados de las cajas.** `min_upward_nz` se queda solo con los píxeles
    cuya normal local apunta hacia arriba —tapas y cubierta—, y es justo lo que hace que
    dos diagonales basten para levantar el mapa: los laterales que ve una son ruido que
    taparía el hueco que ve la otra.

Una celda que nadie vio guarda altura 0 y `observed=False`. **No es lo mismo que cubierta
libre**, y quien lo confunda apilará sobre un hueco que no ha visto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from placing.heightmap import HeightMap, empty_height_map


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole, sin distorsión. `fx`/`fy`/`cx`/`cy` en píxeles."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fovy_deg: float


@dataclass(frozen=True)
class CameraPose:
    pos: np.ndarray                 # [x, y, z] de la cámara, en el mundo
    rot: np.ndarray                 # 3x3, de cámara a mundo


@dataclass(frozen=True)
class DepthFrame:
    """Un fotograma de profundidad con su pose. Es lo que cruza la frontera."""

    depth: np.ndarray               # (H, W), metros desde el plano de la cámara
    intrinsics: CameraIntrinsics
    pose: CameraPose


def unproject_depth_grid(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    depth_min: float = 0.05,
    depth_trunc: float = 8.0,
) -> np.ndarray:
    """La profundidad como imagen de puntos del mundo (H, W, 3). Inválidos a NaN."""
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"la profundidad {(height, width)} no cuadra con las intrínsecas "
            f"{(intrinsics.height, intrinsics.width)}"
        )
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_trunc)
    z = np.where(valid, depth, np.nan)
    x_cam = (cols - intrinsics.cx) * z / intrinsics.fx
    y_cam = -(rows - intrinsics.cy) * z / intrinsics.fy   # fila 0 arriba: Y se invierte
    z_cam = -z                                            # la cámara mira por −Z
    cam_pts = np.stack([x_cam, y_cam, z_cam], axis=-1)
    world = np.tensordot(cam_pts, pose.rot.T, axes=([2], [0])) + pose.pos
    world[~valid] = np.nan
    return world


def upward_facing_mask(points_hw3: np.ndarray, min_upward_nz: float) -> np.ndarray:
    """Los píxeles cuya normal local mira a ±Z del mundo: tapas y cubierta."""
    pts = np.asarray(points_hw3, dtype=np.float64)
    vx = np.full_like(pts, np.nan)
    vy = np.full_like(pts, np.nan)
    vx[:, :-1] = pts[:, 1:] - pts[:, :-1]
    vy[:-1, :] = pts[1:] - pts[:-1]
    normals = np.cross(vx, vy)
    nz = normals[:, :, 2]
    norm = np.linalg.norm(normals, axis=2)
    with np.errstate(invalid="ignore"):
        nz_unit = nz / np.maximum(norm, 1e-12)
        keep = np.abs(nz_unit) >= min_upward_nz
    keep &= np.isfinite(pts[:, :, 0])
    return keep


def unproject_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    depth_min: float = 0.05,
    depth_trunc: float = 8.0,
    min_upward_nz: float | None = None,
) -> np.ndarray:
    """La nube de puntos del mundo (N, 3) de un fotograma de profundidad."""
    grid = unproject_depth_grid(depth, intrinsics, pose, depth_min, depth_trunc)
    if min_upward_nz is not None:
        pts = grid[upward_facing_mask(grid, min_upward_nz)]
    else:
        pts = grid[np.isfinite(grid[:, :, 0])]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def rasterize_points(
    points_xyz: np.ndarray,
    origin_xy: tuple[float, float],
    length: float,
    width: float,
    resolution: float,
    z_min: float,
    z_max: float,
) -> HeightMap:
    """Los puntos, al cajón de su celda, quedándose con el más alto.

    `length` va por X (`axis=1` de la rejilla) y `width` por Y (`axis=0`), igual que en
    `placing.HeightMap`.
    """
    height_map = empty_height_map(
        length=length, width=width, resolution=resolution, origin_xy=origin_xy
    )
    if points_xyz.size == 0:
        return height_map
    pts = np.asarray(points_xyz, dtype=np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ox, oy = origin_xy
    inside = (
        (x >= ox) & (x < ox + length)
        & (y >= oy) & (y < oy + width)
        & (z >= z_min) & (z <= z_max)
    )
    if not np.any(inside):
        return height_map
    i = np.floor((x[inside] - ox) / resolution).astype(np.int32)
    j = np.floor((y[inside] - oy) / resolution).astype(np.int32)
    ny, nx = height_map.grid_shape
    in_grid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    i, j = i[in_grid], j[in_grid]
    z_valid = z[inside][in_grid]
    acc = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(acc, (j, i), z_valid)
    observed = np.isfinite(acc)
    heights = np.zeros((ny, nx), dtype=np.float32)
    heights[observed] = acc[observed].astype(np.float32)
    height_map.heights = heights
    height_map.observed = observed
    return height_map


def fuse_max(maps: list[HeightMap]) -> HeightMap:
    """La superficie observada más alta de cada celda. `observed` es el OR."""
    if not maps:
        raise ValueError("fuse_max necesita al menos un mapa")
    ref = maps[0]
    for other in maps[1:]:
        if other.grid_shape != ref.grid_shape or other.resolution != ref.resolution:
            raise ValueError("los mapas tienen que compartir rejilla y resolución")
        if other.origin_xy != ref.origin_xy:
            raise ValueError("los mapas tienen que compartir origen")
    acc = np.full(ref.grid_shape, -np.inf, dtype=np.float64)
    observed = np.zeros(ref.grid_shape, dtype=bool)
    for hmap in maps:
        observed |= hmap.observed
        acc = np.where(hmap.observed, np.maximum(acc, hmap.heights), acc)
    heights = np.zeros(ref.grid_shape, dtype=np.float32)
    heights[observed] = acc[observed].astype(np.float32)
    return HeightMap(
        origin_xy=ref.origin_xy,
        length=ref.length,
        width=ref.width,
        resolution=ref.resolution,
        heights=heights,
        observed=observed,
    )
