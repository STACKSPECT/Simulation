# Simulation

Paletizado en MuJoCo: un brazo coge paquetes de una cinta que se para, los mide, decide
dónde van y los apila en un palé. La física es real y las métricas se miden, no se
estiman. Todo sale en vivo en la plataforma de observabilidad.

Tres capas —**visión**, **planificación**, **ejecución**— más la medida y la
trazabilidad. Cada una es una carpeta y una frontera; el idioma común está en
`src/contracts.py`.

```
src/vision/       lo que se ve en la cinta, y lo que se mide con el paquete en la mano
src/planner/      mapa de alturas del palé y heurística de score: dónde va
src/cell/         MuJoCo: escena, brazo, cinta, cámaras
src/measure.py    error, apoyo, vuelo, centro de gravedad, margen de estabilidad
src/telemetry.py  la única frontera con la plataforma
src/episode.py    el director: cose los tres módulos
```

## Arrancar

```bash
bash scripts/setup.sh          # venv + dependencias + SDK + modelo del brazo
source .venv/bin/activate

python scripts/palletize.py --viewer            # verlo
python scripts/palletize.py -n 3                # 3 episodios; SUBE por defecto
python scripts/palletize.py -n 3 --no-telemetry # sin subir, solo disco
python tests/test_pallet.py                     # comprobaciones, sin simulador ni red
```

## Estado

Esqueleto. Los contratos están fijados y el documento de arquitectura está escrito; la
implementación de cada módulo está por hacer, y lo que hay que portar del repo
predecesor está marcado fichero a fichero.

**Lee [AGENTS.md](AGENTS.md) entero antes de escribir código.** No es documentación de
cortesía: lleva los tres vocabularios cerrados del contrato con la plataforma y las
trampas que cuestan una ejecución perdida cada una.
