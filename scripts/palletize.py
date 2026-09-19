#!/usr/bin/env python3
"""
El punto de entrada: corre episodios y los sube.

SIN IMPLEMENTAR. Portar el esqueleto desde
`~/HackSpain/paletizado-guionizado/scripts/palletize.py`.

**Aquí es donde se eligen las piezas.** Es el único sitio del repo que sabe si se está
usando la percepción de verdad o su stub-oráculo, y por eso es el único que puede
calcular `oracle`:

    detector = OracleDetector() if args.oracle_vision else CameraDetector()
    gauge    = OracleGauge()    if args.oracle_gauge  else WristGauge()
    planner  = GridPlanner(cfg) if args.naive_planner else ScorePlanner(cfg)
    oracle   = any([args.oracle_vision, args.oracle_gauge, args.naive_planner])

Cambiar de stub a implementación es cambiar una bandera. Eso es lo que permite que los
cuatro módulos avancen en paralelo, y `oracle` mal puesto invalida justo la comparación
que justifica el trabajo: la interfaz no compara un run con oráculo contra uno sin él.

Tres cosas que este fichero tiene que hacer sí o sí:

**Imprimir en qué modo va la telemetría.** El SDK trata "sin credenciales" como su modo
normal, así que un `.env` mal puesto se traga la subida en silencio. En el repo anterior
eso costó varias ejecuciones enteras: se corrían, se miraba la interfaz, no había nada y
no había forma de saber por qué.

**Cerrar SIEMPRE el episodio abierto.** Un Ctrl-C, cerrar el visor o un proceso muerto
dejan la fila en `running` para siempre, y la pantalla Live elige el primer episodio en
ese estado sin ordenar: un solo huérfano la deja clavada ahí indefinidamente.

    try:
        for seed in semillas:
            log.begin(seed, n_objects=...)
            ...
            log.end(result)
    finally:
        if log.episode_id:
            log.end(EpisodeResult(..., success=False, failure=None))
        log.close()

`failure=None` es correcto: no hay causa en el vocabulario que describa "se cerró la
ventana", y `success=False` basta para que quede marcado.

**Pasar `config=run_config(scene)` al `RunLog`**, con su `pallet_size_m`. Sin él la
interfaz supone un europeo de 1200x800 y, con una maqueta, todas las cotas salen mal por
el factor de escala.

Banderas: `--viewer`, `-n N`, `--no-telemetry`, más las tres de elección de piezas.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("entrada sin implementar")


if __name__ == "__main__":
    main()
