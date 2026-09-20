# Migrar stable_pallet a la celda UR10e con tres fuentes

## Qué trae este PR

Este cambio convierte el esqueleto de `Simulation` en una celda autónoma de descarga y
paletizado con UR10e y ventosa OnRobot VGP20. La ejecución admite once niveles —mesa,
cinta y camión, tres por tarea— definidos en `configs/pallet.yaml`.

- `src/cell/` construye la celda en MJCF, controla el brazo y la ventosa, renderiza las
  cuatro vistas admitidas y expone un único contrato `Supply.present()/release()` para
  las tres fuentes.
- `src/episode.py` cose percepción, agarre, medida en muñeca, mapa de alturas,
  planificación, colocación, reasentamiento y medida completa del palé.
- `src/measure.py` publica la medida geométrica y de estabilidad. El CoG usa el
  `cog_offset_m` medido y también incluye cajas que tocan el palé aunque hayan quedado
  fuera de tolerancia.
- `src/telemetry.py` conserva las filas verificadas de `Guionized-simulation` y añade
  fuente, nivel, catálogo, tamaño real del palé y desglose del score.
- `src/vision/oracle.py` aporta los dos oráculos mínimos. `src/planner/naive.py` contiene
  tanto la rejilla oráculo como el beam search real con lookahead.
- `scripts/palletize.py` es el único entrypoint de episodios y siempre cierra run y
  episodio, también al interrumpirlo.
- El panel ofrece dos modos visibles: EJECUCIÓN llama al entrypoint y publica si existen
  credenciales; DEPURACIÓN conserva el runner local y nunca abre telemetría.
- El demostrador original, sus escenarios y su suite completa viven en `tools/` como
  referencia ejecutable.

La escena sigue generándose como una cadena MJCF. El origen ya compone la celda, las
mallas de Menagerie, la ventosa, la carga dinámica y los visuales de fallback de esta
forma; cambiar a `MjSpec` en esta migración duplicaría el constructor y dificultaría
recrear episodios mientras el visor está abierto. MuJoCo compila el mismo modelo final y
AGENTS.md permite explícitamente MJCF.

## Qué queda fuera

No se modifican las fronteras asignadas a visión y planificación:
`src/vision/detect.py`, `src/vision/gauge.py`, `src/planner/heightmap.py`,
`src/planner/heuristic.py` y `src/contracts.py`.

La báscula de muñeca completa (`com_probe.py`, `com_estimator.py`, `tare.py`,
`wrench.py`), el ensayo de transporte y la viga, el benchmark, el generador, los
marcadores de CoM y la reproducción permanecen en `tools/stable_pallet/`. Son ofertas
para sus respectivos módulos, no sustituciones impuestas desde ejecución. El indicador
publicado usa `theker_telemetry.stability_margin`; `validate_stack` continúa siendo una
herramienta interna del planificador y no crea una tercera definición del KPI.

`BeamPlanner` construye temporalmente su estado medido a partir de las colocaciones y
acepta el `Heightmap` del contrato sin depender de su contenido. Esta costura desaparece
cuando el equipo propietario implemente `src/planner/heightmap.py`.

## Calibraciones que no se deben ajustar a ojo

- Palé europeo real: 1,20 × 0,80 m, escala 1.
- Tolerancia de alcance: 20 mm, por encima del suelo medido del servo.
- Corrección cartesiana: tres pasadas después de la solución IK.
- Tránsito: cartesiano a cota fija; no se sustituye por interpolación articular.
- Separación de ventosa: 1,5 mm; holgura entre cajas de una capa: 40 mm.
- Sobre del catálogo: 18–55 × 16–40 × 8–30 cm, 90–280 kg/m³ y hasta 8,5 kg.
- Bahía: 0,74 × 0,88 m; el barrido documentado deja 0 de 720 poses fuera de alcance.
- El brazo colisiona con la celda pero no consigo mismo ni con su propia herramienta.

Cada cifra está junto a su unidad y su procedencia en `configs/scene.yaml` o
`configs/pallet.yaml`.

## Verificación realizada

- `python tests/test_pallet.py`: contratos, vocabularios, niveles, CoG y oráculos.
- `python -m src.measure`: geometría, estabilidad y CoG descentrado.
- Episodio headless de mesa N1: 4/4 intentos completados y artefactos escritos a disco.
- Beam search y rejilla ejecutados contra una escena real de MuJoCo.
- `uv run pytest` en `tools/`: 168 pruebas del demostrador tras la mudanza.

La comprobación física completa de `tests/test_cell.py` queda por repetir después del
último ajuste de la estación de cinta. Las verificaciones remotas de telemetría y del
estado en vivo requieren `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` en
`~/projects/STACKSPECT/.env`; no se falsean si ese fichero no está disponible.

## Qué queda a medias

Además de las fronteras ajenas citadas arriba, faltan la calibración fina de fricción y
consigna de la cinta bajo cargas adversariales, la validación remota con Supabase y una
pasada manual por las once tarjetas en ambos modos del panel. El contrato, los niveles
y los caminos de ejecución quedan preparados para hacer esas comprobaciones sin cambiar
la arquitectura.
