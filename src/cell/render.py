"""
Las vistas del palé que se suben a la plataforma.

El renderer se crea y se destruye en cada llamada: esto pasa unas pocas veces por
episodio y no compensa mantener vivo un contexto GL entre medias. En percepción es al
revés —se renderiza en cada paquete— y por eso vive en otro sitio.

── `view` ES UN VOCABULARIO CERRADO ─────────────────────────────────────────────

`top | side | iso | camera`, y nada más. Con otro nombre el PNG sube a Storage y
**luego** la base rechaza la fila con un `23514`: la foto queda huérfana en el bucket y
la traza se queda sin imagen. Al repo anterior le pasó exactamente con `front`.

Por eso la comprobación está en dos sitios: aquí, y en `tests/test_pallet.py`, que la
hace sin arrancar el simulador. Las cámaras de `configs/pallet.yaml` tienen que llamarse
con uno de esos cuatro nombres o la escena ni siquiera compila.

── Y LO DE SIEMPRE CON LAS FOTOS ────────────────────────────────────────────────

**El brazo se aparta primero, y en CARTESIANO.** Acaba justo encima del palé, que es
donde estaba soltando, así que sin apartarlo la cenital sale del dorso de la mano. Y
apartarlo en espacio de juntas deja el recorrido sin controlar y barre el montón recién
colocado: medido en el repo anterior, el episodio pasaba de 10/10 a 4/10 con
`overhang_violation` en cuanto se metió la foto por capa.

La maniobra vive en `src/episode.py`; aquí sólo se dispara el obturador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Vocabulario cerrado de `snapshots.view`. Ampliarlo obliga a tocar el CHECK de
# `backend/sql/003_snapshots.sql` en Platform: pídeselo a quien lleve el backend.
VIEWS = ("top", "side", "iso", "camera")


@dataclass(frozen=True)
class Snapshot:
    """Una foto tomada, lista para subir."""

    view: str
    after_seq: int
    image: np.ndarray


def render(scene, view: str) -> np.ndarray:
    """Un fotograma RGB desde la cámara `view`, con el palé ya asentado.

    Devuelve el array; convertir a PNG es cosa de `src/telemetry.py`, que es el único
    módulo que sabe que la plataforma existe.
    """
    if view not in VIEWS:
        raise ValueError(
            f"vista {view!r} fuera del vocabulario {VIEWS}. Con otro nombre el PNG sube "
            f"a Storage y luego la base rechaza la fila con un 23514."
        )
    spec = scene.cfg["cameras"][view]
    width, height = spec["resolution"]

    renderer = scene.mujoco.Renderer(scene.model, height=height, width=width)
    try:
        renderer.update_scene(scene.data, camera=view)
        return renderer.render()
    finally:
        # Sin esto el contexto GL sobrevive a la llamada y, tras unas cuantas capas, el
        # proceso se queda sin contextos. Son pocas fotos: abrir y cerrar sale barato.
        renderer.close()
