"""
El obturador de profundidad: una cámara de la escena -> un `DepthFrame` posado.

Es el único fichero de percepción que toca MuJoCo, y toca lo mínimo: renderizar la
profundidad y leer dónde estaba la cámara al hacerlo. Las cuentas —desproyectar,
rasterizar, fusionar— viven en `src/vision/surface.py`, que es numpy puro y se prueba sin
arrancar nada. Partirlo así no es cosmética: es lo que permite que un fallo de fusión se
reproduzca con un array sintético en vez de con un episodio.

Se renderiza una vez por paquete y por cámara, igual que `cell/render.py`: el renderer se
abre y se cierra en cada llamada. Sin el `close()` el contexto GL sobrevive y tras unas
cuantas pasadas el proceso se queda sin contextos.

**Las intrínsecas salen del modelo, no de un fichero.** `fovy` es lo único que MuJoCo
guarda; el resto es un pinhole ideal: píxeles cuadrados, principal en el centro y cero
distorsión. Con una cámara de verdad eso no se cumple y hay que meter la calibración
aquí, que es el sitio donde se nota.

**La profundidad de MuJoCo es Z planar de cámara**, en metros desde el plano de la
cámara. No es longitud de rayo, y `surface.unproject_depth` cuenta con ello.
"""

from __future__ import annotations

import numpy as np

from src.vision.surface import CameraIntrinsics, CameraPose, DepthFrame


def _camera_id(scene, name: str) -> int:
    mujoco = scene.mujoco
    cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise KeyError(f"la cámara {name!r} no está en el modelo")
    return cam_id


def camera_intrinsics(scene, name: str, width: int, height: int) -> CameraIntrinsics:
    """Pinhole ideal a partir del `fovy` que la cámara declara en el modelo."""
    fovy = float(scene.model.cam_fovy[_camera_id(scene, name)])
    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    return CameraIntrinsics(
        fx=float(fy),                    # píxeles cuadrados
        fy=float(fy),
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=int(width),
        height=int(height),
        fovy_deg=fovy,
    )


def camera_pose(scene, name: str) -> CameraPose:
    """Dónde está la cámara y hacia dónde mira, en el mundo, AHORA."""
    scene.mujoco.mj_forward(scene.model, scene.data)
    cam_id = _camera_id(scene, name)
    return CameraPose(
        pos=np.array(scene.data.cam_xpos[cam_id], dtype=np.float64),
        rot=np.array(scene.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3),
    )


def frame(scene, name: str, width: int, height: int) -> DepthFrame:
    """Un fotograma de profundidad de la cámara `name`, con su pose e intrínsecas."""
    renderer = scene.mujoco.Renderer(scene.model, height=height, width=width)
    try:
        renderer.update_scene(scene.data, camera=name)
        renderer.enable_depth_rendering()
        depth = np.asarray(renderer.render(), dtype=np.float32)
    finally:
        renderer.close()
    return DepthFrame(
        depth=depth,
        intrinsics=camera_intrinsics(scene, name, width, height),
        pose=camera_pose(scene, name),
    )
