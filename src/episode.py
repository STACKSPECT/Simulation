"""
El episodio: el director. Cose los tres módulos y mide lo que hace la física.

PORTAR el esqueleto desde `~/HackSpain/paletizado-guionizado/src/pallet/episode.py` —la
maniobra de `_pick`/`_place`/`_release`/`_settle`/`_shoot`, `Episode`, `Snapshot` y el
contador de eventos están resueltos y medidos— y sustituir el guion por las llamadas a
los módulos.

**Este fichero pide, no calcula.** Si aquí aparece aritmética de huecos, de alturas o de
confianza de detección, se ha colado lógica que le toca a otro. Lo único que decide es
el orden y qué hacer cuando algo falla.

El bucle por paquete, que es el orden que define la arquitectura:

    conveyor.present()      la cinta avanza y PARA.  None -> timeout
    detector.observe()      -> [Observation].  vacío -> no_detection
    _pick()                                    no converge -> ik_unreachable
                                               se escurre  -> grasp_slip
    gauge.measure()         -> PackageSpec     (con el paquete YA en la mano)
    heightmap.measure()     -> Heightmap       (MEDIDO, no un contador)
    planner.choose()        -> PlacementPlan   None -> wrong_placement
    _place()                                   y se mide la deriva al asentarse
    conveyor.release()      con el brazo ya retirado
    _record()               mide TODAS las cajas, no solo la nueva

Lo que hay que conservar de la versión guionizada, palabra por palabra:

**El paquete cuenta como INTENTADO desde que se toca**, y deja su fila pase lo que pase.
Si el agarre o el depósito fallan, sale con `placed=false` y el estado del palé queda
como estaba. Sin eso, el intento que tumba el episodio sería el único que no aparece en
la tabla.

**`_record` vuelve a medir todas las cajas.** Lo que avisa de un derrumbe es que la
última descoloque a las de abajo, y eso no se ve mirando solo la que acaba de caer. Una
caja que estaba bien puesta y ahora no lo está ES la definición operativa de derrumbe:
sale de medir, no de un umbral inventado.

**`seq` de los eventos es un contador del EPISODIO entero** (`len(episode.events)`), no
del paquete: hay cinco o seis eventos por paquete y la base exige `unique(episode_id,
seq)`. Un duplicado tumba la fila y, con ella, la subida del resto del episodio.

**`ts` es tiempo SIMULADO**, no de reloj: así dos ejecuciones de la misma semilla dan la
misma línea temporal por rápida que sea la máquina.

**La foto de cada capa se toma con el brazo apartado en CARTESIANO.** En espacio de
juntas el recorrido no está controlado y barre el montón recién colocado: medido, el
episodio pasaba de 10/10 a 4/10 con `overhang_violation` en cuanto se metió la foto por
capa. `park_joints=True` solo para la foto final, cuando ya no queda nada que colocar.

Lo que CAMBIA:

  - `run_episode` recibe el `Detector`, el `Gauge` y el `Planner` —los tres `Protocol`
    de `contracts.py`—, no un guion. Quién es cada uno lo elige `scripts/palletize.py`.
  - El payload de `perceive` deja de ser `{seen: las que faltan, confidence: 1.0}` y
    pasa a ser lo que de verdad detectó y con qué confianza. El estado de la cinta cabe
    ahí también: en el payload sobrar es inocuo.
  - El payload de `plan` lleva el `layer` y el `slot` que decidió el planificador, y de
    propina su `breakdown` del score.
  - "No hay hueco" (`planner.choose()` devuelve `None`) es un `wrong_placement`
    registrado, no una excepción.
"""

from __future__ import annotations

from src.contracts import Detector, Gauge, Planner, Sink


def run_episode(scene, detector: Detector, gauge: Gauge, planner: Planner,
                seed: int = 0, speed: float = 1.0, sink: Sink | None = None):
    """Monta el palé entero y devuelve lo medido."""
    raise NotImplementedError("episodio sin implementar: portar y coser los módulos")
