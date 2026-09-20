from __future__ import annotations

import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Pin the visual model so a demo always renders the same geometry.
MENAGERIE_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
UR10E_MESH_FILES = (
    "base_0.obj",
    "base_1.obj",
    "shoulder_0.obj",
    "shoulder_1.obj",
    "shoulder_2.obj",
    "upperarm_0.obj",
    "upperarm_1.obj",
    "upperarm_2.obj",
    "upperarm_3.obj",
    "forearm_0.obj",
    "forearm_1.obj",
    "forearm_2.obj",
    "forearm_3.obj",
    "wrist1_0.obj",
    "wrist1_1.obj",
    "wrist1_2.obj",
    "wrist2_0.obj",
    "wrist2_1.obj",
    "wrist2_2.obj",
    "wrist3.obj",
)


def _asset_cache() -> Path:
    configured = os.environ.get("STABLE_PALLET_ASSET_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "stable-pallet" / "mujoco-menagerie" / MENAGERIE_COMMIT


def _download_mesh(filename: str, destination: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
        f"{MENAGERIE_COMMIT}/universal_robots_ur10e/assets/{filename}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "stable-pallet/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            payload = response.read()
        if len(payload) < 100:
            raise RuntimeError(f"Downloaded UR10e mesh is unexpectedly small: {filename}")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_ur10e_mesh_assets() -> dict[str, bytes]:
    """Return official UR10e visual meshes, downloading the pinned files once."""
    cache = _asset_cache()
    cache.mkdir(parents=True, exist_ok=True)
    missing = [filename for filename in UR10E_MESH_FILES if not (cache / filename).is_file()]
    if missing:
        try:
            with ThreadPoolExecutor(max_workers=min(6, len(missing))) as executor:
                futures = [executor.submit(_download_mesh, name, cache / name) for name in missing]
                for future in futures:
                    future.result()
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "Could not download the pinned MuJoCo Menagerie UR10e meshes. "
                "Check the network or run with --simplified-graphics."
            ) from exc
    return {filename: (cache / filename).read_bytes() for filename in UR10E_MESH_FILES}
