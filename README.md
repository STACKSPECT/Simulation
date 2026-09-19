# Simulation

Paletizado autónomo en MuJoCo con UR10e y ventosa OnRobot VGP20. La misma celda recoge
bultos preparados en mesa, entregados por una cinta física o apilados en un remolque,
los mide, decide dónde van y los deposita en un europalé. La física y las métricas se
miden y la ejecución se publica en vivo en la plataforma de observabilidad.

Tres capas —**visión**, **planificación**, **ejecución**— más la medida y la
trazabilidad. Cada una es una carpeta y una frontera; el idioma común está en
`src/contracts.py`.

```
src/vision/       lo que entrega la fuente, y lo que se mide con el paquete en la mano
src/planner/      mapa de alturas del palé y heurística de score: dónde va
src/cell/         MuJoCo: escena, brazo, mesa, cinta, camión y cámaras
src/measure.py    error, apoyo, vuelo, centro de gravedad, margen de estabilidad
src/telemetry.py  la única frontera con la plataforma
src/episode.py    el director: cose los tres módulos
```

## Arrancar

```bash
bash scripts/setup.sh          # venv + dependencias + SDK + modelo del brazo
source .venv/bin/activate

python scripts/palletize.py --viewer --source table
python scripts/palletize.py -n 3 --level 23     # cinta N3; SUBE por defecto
python scripts/palletize.py -n 1 --level 31 --no-telemetry
python tests/test_pallet.py                     # contrato, sin simulador ni red
python tests/test_cell.py                       # las tres fuentes y la IK
python -m src.measure                           # geometría y CoG
```

## Estado

La capa de ejecución, la medida, la telemetría, el episodio, los oráculos y el beam
search están implementados. `src/vision/detect.py`, `src/vision/gauge.py`,
`src/planner/heightmap.py` y `src/planner/heuristic.py` siguen siendo las fronteras de
los equipos de visión y planificación. Mientras llegan, el entrypoint usa los oráculos
de visión y medida y el beam search real, y marca el run como `oracle`.

El panel del demostrador ofrece dos modos visibles: **EJECUCIÓN** lanza este entrypoint
y usa telemetría; **DEPURACIÓN** usa el runner local de `tools/` y no sube nada.

**Lee [AGENTS.md](AGENTS.md) entero antes de escribir código.** No es documentación de
cortesía: lleva los tres vocabularios cerrados del contrato con la plataforma y las
trampas que cuestan una ejecución perdida cada una.
