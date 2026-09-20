# AGENTS.md

Contexto permanente de este repositorio. Léelo entero antes de escribir código. Si algo
aquí contradice lo que crees saber, gana este documento; si algo está desactualizado,
corrígelo en el mismo PR.

La documentación de cara afuera está en inglés y en otros dos ficheros: `README.md`
—estado fichero a fichero, dependencias con sus licencias y decisiones abiertas— y
`CONTRIBUTING.md` —ramas, convención de commits y la escalera de comprobación de §9—.
Este documento sigue siendo el que manda sobre el contenido; aquéllos no lo repiten, lo
apuntan. El proyecto es MIT (ver `LICENSE`).

---

## 1. Qué es esto

Una simulación de paletizado **de verdad**: un UR10e con ventosa OnRobot VGP20 coge
paquetes de una mesa, una cinta o un remolque, los mide, decide dónde van y los apila
en un europalé. Hay catorce niveles seleccionables —siete de mesa, tres de cinta y
cuatro de camión—, y los dos últimos de mesa **no se espera que salgan verdes** (§6).
La física es
MuJoCo, las métricas se miden y todo se sube en vivo a la plataforma de observabilidad.

El predecesor —`STACKSPECT/Guionized-simulation`— hacía lo mismo con el hueco de cada
caja escrito en un YAML. Existía para fijar el contrato con la plataforma y recorrerlo
entero antes de que existiera el sistema real. **Este repo es el sistema real.** La
diferencia no es cosmética:

| | guionizado | aquí |
|---|---|---|
| dónde va cada caja | escrito en `configs/pallet.yaml` | lo decide la heurística con lo que ve y lo que mide |
| qué entrega la fuente | nada, las cajas esperan colocadas en la mesa | lo dice la percepción, y puede equivocarse |
| dimensiones del paquete | del catálogo | se miden con el paquete ya en la mano |
| `oracle` del run | `true` | **`false`** en cuanto no queda ningún stub |
| `no_detection` | inalcanzable | alcanzable |

## 2. La arquitectura

Tres capas, más dos transversales. Cada carpeta es una frontera: se toca la de uno sin
abrir la de los demás.

| Capa | Carpeta | Qué hace |
|---|---|---|
| **Visión** | `src/vision/` | Ve el paquete que la fuente presenta (`detect.py`), lo **mide** ya en la mano (`gauge.py`) y **mide la superficie del palé** con tres cámaras de profundidad (`depth.py` renderiza, `surface.py` desproyecta y fusiona, numpy puro) |
| **Planificación** | `src/planner/` | Mapa de alturas del palé (`heightmap.py`) y **heurística de score** que elige el hueco (`heuristic.py`, adaptador sobre `placing/`) |
| **Ejecución** | `src/cell/` | MuJoCo: escena, brazo, fuente —mesa, cinta o camión—, mesa auxiliar común, cámaras y ensayo físico opcional (`scene.py`, `arm.py`, `conveyor.py`, `table.py`, `truck.py`, `render.py`, `stability.py`) |
| Medida | `src/measure.py` | El resultado: error, apoyo, vuelo, CoG del palé, margen de estabilidad |
| Trazabilidad | `src/telemetry.py` | **Única** frontera con la plataforma |

Los cose `src/episode.py`, que es el director: pide, no calcula. Y todos hablan el mismo
idioma, `src/contracts.py`.

El ciclo de un paquete, que es también el orden en que se llaman los módulos:

```
la fuente entrega y PARA     cell.conveyor.Supply.present()
  -> se ve                   vision.detect.observe()      -> Observation
  -> se coge                 cell.arm                     (ejecución)
  -> se mide en la mano      vision.gauge.measure()       -> PackageSpec
  -> se decide el hueco      planner.heuristic.choose()   -> PlacementPlan
  -> se coloca               cell.arm                     (ejecución)
  -> se mide el resultado    measure.measure_placement()  -> Placement, PalletState
  -> se cuenta               telemetry.RunLogSink         (filas en vivo)
```

Fíjate en el orden: **se mide DESPUÉS de coger y ANTES de planificar**. No es un capricho
de diseño, es lo que obliga a que el planificador sea incremental — no puede precalcular
el palé entero porque no sabe cómo es el siguiente paquete hasta que lo tiene agarrado.

## 3. Los contratos

`src/contracts.py` es el único fichero que leen los cuatro módulos, y el único que hay
que consensuar antes de tocar nada. Quién produce qué:

| Dato | Lo produce | Lo consume | Qué lleva |
|---|---|---|---|
| `Observation` | `vision/detect.py` | `episode.py` (para ir a coger) | id, pose en la cinta, dims aproximadas, `confidence` |
| `PackageSpec` | `vision/gauge.py` | `planner`, `telemetry` | `dims_m`, `mass_kg`, `cog_offset_m`, `grasp_width`, `type_name` |
| `Heightmap` | `planner/heightmap.py` | `planner/heuristic.py` | rejilla de alturas del palé, en metros sobre la cubierta, y `observed`: qué celdas vio alguna cámara. Altura 0 sin observar NO es cubierta libre |
| `PlacementPlan` | `planner/heuristic.py` | `episode.py` (dónde soltar) | pose objetivo (centro, en el mundo, yaw en radianes), `layer`, `slot`, `score`, apoyo previsto, `breakdown` de 14 términos |
| `Placement`, `PalletState` | `measure.py` | `telemetry.py` | lo medido: error, apoyo, vuelo, CoG, margen, relleno |

Y cuatro `Protocol` —`Detector`, `Gauge`, `Planner`, `Sink`— que es lo que permite
trabajar en paralelo sin bloquearse.

**La regla que hace que esto funcione: cada módulo entrega su stub-oráculo ANTES que su
implementación.** El stub lee la verdad de la escena (`cell.scene` sabe dónde está cada
caja y cuánto pesa) y devuelve el contrato sin error. Con los cuatro stubs, el bucle
entero corre verde el primer día, y sustituir uno por lo bueno es cambiar **una línea**
en `scripts/palletize.py`. Sin eso, nadie puede probar lo suyo hasta que estén los
cuatro, que es exactamente como se pierde una semana.

Los stubs viven al lado de lo que sustituyen: `vision/oracle.py`, `planner/naive.py`.
`naive.py` contiene además `BeamPlanner`: no es un stub, usa el beam search medido del
demostrador y por tanto no activa `oracle`. Solo `GridPlanner` lo activa.

**`oracle` del run es `any(stub en uso)`.** Un run con el detector oráculo NO es un run
con percepción, y la interfaz no compara los dos: marcarlo mal invalida justo la
comparación que justifica el trabajo. Se calcula en `scripts/palletize.py`, donde se
eligen las piezas, y no se escribe a mano en ningún otro sitio.

## 4. Puesta en marcha

```bash
bash scripts/setup.sh          # venv + dependencias + SDK + modelo del brazo
source .venv/bin/activate

python scripts/palletize.py --viewer            # verlo
python scripts/palletize.py -n 3                # 3 episodios; SUBE por defecto
python scripts/palletize.py -n 3 --no-telemetry # sin subir, solo disco
python scripts/palletize.py --source truck      # primer nivel de camión
python scripts/palletize.py --level 23          # nivel concreto de cinta
python scripts/palletize.py --no-oracle-gauge --no-telemetry --level 11
python scripts/palletize.py --no-oracle-gauge --precise-com --no-telemetry --level 11
python -m placing                               # la heuristica sola: 15 asserts, sin nada
python scripts/palletize.py --no-oracle-heightmap   # el mapa, con camaras
python scripts/palletize.py --beam-planner      # el beam search, para comparar
python scripts/palletize.py --naive-planner     # la linea base en rejilla
python scripts/palletize.py --stability-test    # al final: 15 sacudidas y viga X/Y
python tests/test_pallet.py                     # comprobaciones, sin simulador ni red
python tests/test_cell.py                       # las tres fuentes, cámaras, IK y el gauge
python -m src.measure                           # la medida, con sus asserts
cd tools && uv run --extra dev pytest           # el demostrador sobrevive a la mudanza
```

El SDK `theker_telemetry` **no vive aquí**: es el contrato de lo que la plataforma
guarda, y quien define la forma del dato es quien lo almacena. `setup.sh` lo instala
editable desde el repo Platform y comprueba que trae `RunLog.begin` y
`stability_margin`: **solo la rama `dev`** los tiene. Con `main` el fallo no aparece al
instalar, sino al correr.

Las credenciales de Supabase se leen del `.env` del repo o del de su carpeta padre.
`SUPABASE_SERVICE_KEY` se salta el RLS y es la única que escribe: solo lado Python.

## 5. Reglas duras

Ninguna se deduce leyendo el SDK. Todas costaron una ejecución perdida en el repo
anterior.

**Se sube por defecto, y se dice.** Con `.env` presente la telemetría va activa sin pedir
nada, y al arrancar se imprime en qué modo corre. El SDK trata "sin credenciales" como su
modo normal, así que un `.env` mal puesto se traga la subida en silencio: se corría, se
miraba la interfaz, no había nada y no había forma de saber por qué.

**Un episodio abierto se cierra SIEMPRE.** Un Ctrl-C, parar desde el panel, cerrar el
visor o un proceso muerto dejan la fila en `running` para siempre, y la pantalla Live
elige el primer episodio en ese estado **sin ordenar**: un solo huérfano la deja clavada
ahí indefinidamente. El bucle va en `try/finally` y en el `finally` se cierra con
`success=False, failure=None`. `SIGTERM` no recorre ese `finally`: el panel pide `SIGINT`
y el CLI trata `SIGTERM` como Ctrl-C.

**El disco manda.** El `episodes.jsonl` de `runs/` es la fuente de verdad y Supabase una
réplica. Un fallo de red avisa una vez, apaga la subida y el episodio sigue. **No
envuelvas las llamadas de telemetría en try/except**: el SDK ya lo hace, y hacerlo dos
veces esconde el aviso.

**Se escribe en vivo, no de golpe.** `begin()` abre el episodio en curso,
`event()`/`placement()`/`pallet_state()`/`snapshot()` sueltan filas según se miden,
`end()` lo cierra con un `PATCH`. Ése es todo el motivo por el que la pantalla Live está
viva. **No mezcles `episode()` con `begin()`** en el mismo episodio: insertaría la fila
dos veces y el `unique(run_id, seed)` tumba la segunda. `episode()` es para el backfill.

**Las columnas van en SI y los `payload` de eventos en mm y grados.** Contradice la regla
general del proyecto y es la incoherencia que más fácil se cuela.

```
perceive  {seen, confidence}          pick    {mass_kg, type}
plan      {layer, slot}               settle  {layer, drift_mm}
place     {error_xy_mm, error_yaw_deg, overhang_mm}
fail      {cause}
```

Y la asimetría que hay que tener clarísima:

- **En `payload` (que es `jsonb`), sobrar es inocuo y faltar rompe.** Puedes añadir
  `score` al `plan` o `cog_offset_mm` al `pick` sin miedo. Pero si falta una de las
  claves de arriba, la interfaz **no enseña nada y no da error**.

  Lo que esta celda manda hoy además del mínimo, y por qué: `perceive` lleva el estado de
  la fuente (`source`, `pending`, `exhausted`, `jammed`) porque ninguna fuente tiene
  evento propio (ver §6); `plan` lleva `score`, `breakdown` y `heightmap_top_mm`; `pick`
  lleva `cog_offset_mm`, `active_cups` y `grip_capacity_ratio` —el agarre pierde margen
  antes de resbalar, así que un `grasp_slip` se ve venir en el ratio—; y `place` lleva
  `reach_residual_mm`, lo que el brazo NO llegó a corregir. Ese último distingue un
  acierto limpio de un roce justo por debajo de `reach_tolerance`, que sin él acaban los
  dos en `True`. Ninguno lo pinta la interfaz todavía: están guardados y consultables.
- **En el nivel de la fila, sobrar es letal.** `event/placement/pallet_state/snapshot`
  son `**kwargs` puros y cada clave es una columna: un nombre mal escrito es un 400 que
  el SDK se traga, y a partir de ahí la subida queda apagada **para el resto de la
  ejecución**.

**`seq` es único por episodio, no por paquete.** Hay cinco o seis eventos por paquete. Un
contador del episodio entero (`len(episode.events)`) lo resuelve; el índice del paquete,
no. Un duplicado tumba la fila y, con ella, la subida del resto del episodio.

**Una fila de `pallet_states` por paquete INTENTADO**, incluido el que derrumba el montón
y los que fallaron antes de depositar. Es justo la que hace que la traza de CoG cruce el
cero. Si solo emites los que salieron bien, la traza acaba en verde en un episodio que se
cayó y el gráfico deja de servir para lo único que existe: dejar ver venir el fallo
varias colocaciones antes.

**El CoG cuenta las cajas fuera de tolerancia.** Una caja mal puesta sigue pesando y sigue
moviendo el centro de gravedad. Lo que NO cuenta es la que se quedó en la mesa: el
criterio es geométrico —¿su huella toca el palé?—, no "¿falló la maniobra?". Ver
`measure.on_pallet`.

**El margen de estabilidad se mide contra el polígono de soporte**, la envolvente de las
huellas de la capa 1, no contra el borde del palé. Contra el palé los números salen
optimistas y la pantalla dice que todo va bien hasta el derrumbe.

**Cuatro vocabularios cerrados.** Inventar un valor no da error donde lo escribes:

- `events.kind`: `perceive | plan | pick | place | settle | fail`. **No hay kind para la
  cinta** (ver §6).
- `snapshots.view`: `top | side | iso | camera`. Con otro nombre el PNG sube a Storage y
  **luego** la base rechaza la fila con un `23514`: la foto queda huérfana y la traza sin
  imagen. Al repo anterior le pasó con `front`.
- `failure`: `no_detection`, `ik_unreachable`, `collision`, `grasp_slip`,
  `wrong_placement`, `timeout`, `stack_collapse`, `overhang_violation`. `EpisodeResult`
  lanza `ValueError` con cualquier otro, a propósito. Añadir uno obliga a tocar tres
  sitios en Platform: pídeselo a quien lleve el backend, no lo inventes aquí.
- `levels[].decor`: `cell | plant`. Éste no lo valida la plataforma —no sube como
  columna, va dentro de `runs.config`— pero lo valida `scene.levels()`, y por la misma
  razón que los otros tres: `row.get("decor", "cell")` se traga cualquier nombre y la
  escena saldría con el decorado por defecto sin decir nada. Ver §6.

Y `task` es **la fuente del nivel**: `table`, `conveyor` o `truck` — mesa, cinta y camión,
que es lo que la interfaz enseña traducido. Sale de `scene.level.source` y no se escribe a
mano: así las tres tareas se comparan entre sí en vez de ser todas «paletizado». El SDK lo
valida contra `TASKS`, y el CHECK de `runs`/`episodes` acepta además el histórico
(`induction`, `palletizing`), que es lo que ya está subido.

**`synthetic` no lo escribe esta simulación** (es para datos sembrados), `git_sha` y
`oracle` van en el run y no se repiten por episodio, y `status` sale de
`EpisodeResult.success`: no lo calcules.

**`config.pallet_size_m` es obligatorio.** La interfaz dibuja a escala real y
`palletSize()` lo busca en `runs.config` primero. Aquí el palé es un europeo real de
1,20 × 0,80 m y `scale` es 1. Sin el campo la coincidencia sería accidental y una
variante a otra escala volvería a dibujarse mal. En `config` cabe la fuente, el nivel,
el catálogo y las velocidades de la celda.

**Las fotos, una por capa, y con el brazo apartado en CARTESIANO.** El brazo acaba justo
encima del palé, que es donde estaba soltando: sin apartarlo la cenital sale del dorso de
la mano. Y apartarlo en espacio de juntas deja el recorrido sin controlar y barre el
montón recién colocado: medido en el repo anterior, el episodio pasaba de 10/10 a 4/10
con `overhang_violation` en cuanto se metió la foto por capa. El `after_seq` de la foto
casa con el de la traza de CoG: la imagen y ese punto del gráfico son el mismo instante.

## 6. Lo nuevo respecto al guionizado

### La percepción del palé, y las seis cámaras

El mapa de alturas se MIDE con tres cámaras que miran al palé: la cenital y dos
diagonales opuestas. Se renderiza profundidad, se desproyecta a puntos del mundo, se tira
todo lo que no mire hacia arriba (`min_upward_nz`: tapas y cubierta sí, costados no), se
rasteriza sobre la huella del palé y se fusiona quedándose con lo más alto de cada celda.

**En MJCF hay seis cámaras y en el vocabulario de la plataforma hay cuatro, y eso es a
propósito.** `snapshots.view` es un CHECK cerrado —`top | side | iso | camera`— y sólo
las de `cameras:` suben fotos. Las dos diagonales viven en `perception.rig:`, otro bloque,
porque no suben ninguna: si se declararan ahí abajo, `_cameras_xml` las rechazaría, y si
se colaran, el PNG subiría a Storage y la base rechazaría la fila con un 23514.

**La cenital es la MISMA `top` que ve la plataforma**, compartida a propósito. Dos
cámaras cenitales que se van separando con los años acaban en una foto y un mapa que no
se corresponden.

Tres avisos:

- **`observed` no es decoración.** Una celda que ninguna cámara vio guarda altura 0, que
  es indistinguible de cubierta libre. No lo son. `allow_unobserved` decide qué hacer con
  ellas, y arranca en `true` porque la cobertura de este rig **no está medida**: sobre un
  palé vacío da 93.3 %, no el 98 % que haría razonable bajarlo a `false`. Las posiciones
  de `perception.rig` son de arranque, escaladas del palé de 0.60x0.40 de la feature.
- **Las cámaras ven el brazo.** Sobre un palé vacío con el brazo en casa, el `top` del
  mapa sale 1.18 m: no es el montón, es el propio robot cruzando el encuadre. A mitad de
  episodio el brazo está en la estación y no estorba, pero el problema está ahí y lo que
  toca es excluir los geoms del robot del render de profundidad, no subir `z_max`.
- **El mapa oráculo tiene que contar la MISMA verdad.** `measure_ground_truth` estampa
  **dos mesas nombradas, no cualquiera que invada**: la auxiliar siempre, y la de recogida
  sólo en los niveles de mesa, que es donde se monta. Una tercera mesa futura no se
  estamparía sola: hay que añadirla a esa lista (`heightmap._stamp_static_obstacles`).
  Hoy la de recogida deja 60 mm de aire y la auxiliar común deja 50 mm, así que ninguna
  pisa una sola celda —lo ancla `tests/test_pallet.py`, sin escena—, pero el estampado
  sigue siendo obligatorio si se mueve alguna: sin él, el oráculo ofrece una esquina
  ocupada que las cámaras sí ven. Ya se midió ese fallo con la mesa antigua: el brazo
  empujaba la caja contra ella, se quedaba a 135 mm del destino y el episodio moría en
  `ik_unreachable`.

### Las fuentes

`cell.conveyor.Supply` expone `present()` y `release()` para tres implementaciones. El
camión presenta la carga completa y elige siempre la caja más alta, la única que no
sostiene otra. **La mesa y la cinta son la MISMA banda**, y lo único que las distingue es
dónde acaba el carril: `scene.lane_for` se lo da a cada una, y por eso `TableSupply` es un
`Belt` sin cuerpo. `release()` se llama cuando la mano ya no está encima de la fuente.

**Los bultos salen de una caja negra, y eso no es decorado.** Los que esperan se aparcaban
en fila —`x = -3 - i*0.7`, o sea 21 m con los treinta del nivel 17— y se veían salir de la
escena hacia el infinito. Ahora esperan en rejilla dentro de un cerramiento opaco que se
dimensiona solo con la carga (`scene.parking_grid` calcula la rejilla y el cerramiento se
ajusta a ella: son la misma cuenta a propósito, porque si cada uno la hiciera por su lado
los bultos aparecerían atravesando la pared). Treinta bultos ocupan ahora 3,00 m en vez de
21. Es además lo que la ficción dice: el sistema no sabe lo que viene.

**La banda tiene dos tramos físicos contiguos y un ascensor antes de la entrada.**
`Belt` carga cada caja abajo, eleva la plataforma mocap por una rampa suave y sólo
entonces arranca la cinta. La caja sube por contacto: no se reescribe su pose durante
el trayecto. El ascensor baja vacío antes de cargar la siguiente y su posición sobrevive
a `rebuild()`. El plazo de entrega incluye bajada, subida, transporte y reposo; un fallo
sigue siendo `timeout`. La prueba de transporte en `tests/test_cell.py` comprueba todo
el recorrido y dos entregas consecutivas, con ambos modos gráficos y ambas fuentes.

Tres cosas de la banda de mesa que no se deducen leyendo el código:

- **Acaba en el canto de la mesa y comparte su cota**, las dos a 0.58. `Belt._riding` mira
  la ALTURA para decidir a quién arrastra, así que la banda sigue empujando ya sobre la
  mesa y para el bulto en su centro. Medio centímetro de escalón y deja de empujar justo
  al llegar: el episodio muere en `timeout` y no se ve por qué. Hay un test sin simulador
  que ancla los tres números.
- **El centro de la banda se DERIVA del canto de la mesa**, no va escrito. Dos números que
  hay que cuadrar a mano acaban descuadrados.
- **Cuesta tiempo simulado**: el nivel 11 pasó de 70,4 s sólo con banda a 99,0 s con
  ascensor (semilla 1, `speed=1`). La subida de 0,54 m tarda 2,7 s y la bajada vacía,
  otros 2,7 s; también se añade recorrido horizontal antes de la cinta. La prueba del
  ciclo deja 115 s de presupuesto. La fuente del nivel 17 entrega sus treinta paquetes
  en 424,4 s sin maniobras del brazo: ese tiempo también cuenta en `max_duration_s`.

Además de la fuente, **los doce niveles montan una mesa auxiliar vacía** a la derecha
del palé. No entrega paquetes ni cambia `Supply`: es una superficie física común donde
el robot puede apartar uno si una estrategia lo necesita. **Hoy no la consume ningún
camino del código**; se acepta a propósito como superficie disponible. Deja **100 mm de
aire** hasta el palé para no contaminar su medida — y eso es más importante de lo que
suena: su cara superior está 436 mm POR ENCIMA de la cubierta, así que desde el punto de
vista del brazo es una pared, no una mesa. Con los 50 mm que tenía antes, la herramienta
bajaba pegada a ella sobre los huecos de canto —que son justo los que mejor puntúa el
planificador— y tiraba la carga: nivel 33, semilla 1, `--speed 1`, la cuarta caja medida
a 1674 mm y `stack_collapse`. **En fast-forward no se ve**, porque el brazo teletransporta
entre waypoints; el visor y el panel sí corren el tramo que roza. Ver la cabecera de
`auxiliary_table` en `configs/scene.yaml`, que trae el barrido entero.

Lo comprobado de esa mesa es **su centro, no su huella**: cabe ahí el bulto máximo
girado, y `tests/test_cell.py` deja un bulto reposando en él de verdad. Parte de la
huella queda fuera del alcance del UR10e —68 de 81 celdas a la cota de suelta, 54 de 81
a la de aproximación—, así que una estrategia que quiera soltar fuera del centro tiene que repetir
antes el barrido de IK. Las cifras y el barrido están en `configs/scene.yaml`.

Dos avisos:

- **Ninguna fuente tiene evento propio.** `events.kind` es un CHECK cerrado de seis valores
  y ninguno es suyo. Su estado viaja en el `payload` de `perceive` (donde sobrar es
  inocuo) o no viaja. No inventes un kind: la fila la rechaza la base y se apaga la
  subida del resto del run.
- **Un atasco se reporta como `timeout`.** No hay causa de fallo para "la fuente no
  entregó". Si la estación se queda vacía y expira la espera, es `timeout`; si el
  cartón llega y no se asienta, también. Si entrega pero la percepción no ve nada, es
  `no_detection`. Son cosas distintas y conviene no mezclarlas, porque el gráfico de
  fallos las separa. En la cinta, PARA significa reposo medido: `present()` no
  devuelve tras un settle fijo.
- **La estación no puede estar en el canto de la banda.** El cartón para donde le dicen,
  y si eso lo deja con medio cuerpo en el aire vuelca, retrocede o se cae — no falla la
  ventosa, falla la geometría. Por delante de la estación tiene que quedar al menos la
  semihuella GIRADA del bulto más largo del catálogo. Las cifras, en `configs/scene.yaml`.

### La envolvente del catálogo, que es más estrecha de lo que parece

Los ocho tipos de `configs/pallet.yaml` no son un surtido: son **casi exactamente** lo
que esta celda sabe manipular. Salirse por cualquiera de los dos extremos falla, y falla
tarde. Medido añadiendo los niveles 14 y 15; las tablas están en la cabecera del catálogo.

- **`book_s` (0.24 × 0.18) ya es la caja más pequeña que se puede DEJAR.** No lo limita el
  agarre —las ventosas sobran— sino el cuerpo del VGP20: mide 264 × 184 mm y la holgura
  del planificador son 40 mm *alrededor de la caja*, no de la herramienta. Con la caja más
  corta que 264 − 2·40 = 184 mm el anillo deja de tapar al cuerpo, la caja entra en su
  hueco y la herramienta pisa al vecino al bajar. Error al soltar junto a un vecino:
  89,4 mm con 0.14 × 0.11, 22,9 mm con 0.19 × 0.13, 2,1 mm con 0.24 × 0.18.
- **Nada por debajo de 0.10 de alto.** Una caja de 0.06 deja el TCP a 0.205 m al depositar
  sobre la cubierta, y el canto del palé más cercano al pedestal cae a 0.42 m de él: justo
  la banda donde el brazo no SOSTIENE la pose. Con `reach_max: 0` el filtro de alcance está
  apagado, así que el planificador elige ese hueco igual y el episodio muere en
  `ik_unreachable`.
- **Lo que importa de una mezcla no es cuántos bultos tiene, sino qué fracción mide 0.10
  de alto**, porque cada uno de ésos es un dado contra esa banda. Con 12 bultos y 8
  semillas: 9 bajos de 12 dan 2/8 episodios en verde; 3 bajos de 12 dan 5/8. Reparte las
  alturas como están repartidas las de los ocho originales.
- **Cuadrar las alturas en cursos NO compensa.** Ayuda al apoyo —`min_support_ratio` 0.60
  pide el 60 % de la huella apoyada a una misma altura, y un montón con doce bandas deja a
  una caja grande sin una sola ventana válida con el palé al 44 %— pero amontona
  colocaciones en la banda muerta y sale más caro de lo que ahorra.

Los dos techos son **preexistentes**: se ven en los niveles nuevos sólo porque colocan más
bultos que ningún otro. El nivel 13, que ya estaba, va 3/5 en el mismo barrido de semillas.
Los niveles de este repo están verdes **en la semilla por defecto, no en todas**. Subirlos
exige medir el alcance del brazo CON CARGA y encender `reach_max`/`reach_min` —trabajo
aparte, y toca los niveles que ya existen—, no retocar la mezcla.

### Los niveles 16 y 17 están en rojo A PROPÓSITO

Los once primeros están dimensionados para que la heurística los apruebe. **El 16 y el 17
no**: son el escalón que `placing` no alcanza, y existen para medir cuánto le falta. Un
rojo ahí es el resultado del experimento, no una configuración rota — **no los arregles
bajándoles el número de bultos.** Si alguna vez salen verdes sin que nadie los toque, eso
es la noticia: algo mejoró de verdad.

Línea base medida con `ScorePlanner`, oráculos de visión y mapa, semillas 1-6, **antes
del ascensor físico**. Su tiempo adicional puede cambiar la estabilidad de la pila:

| nivel | bultos | media colocada | cómo se rompe |
|---|---|---|---|
| 16 | 20 | 10,3 | 6/6 se queda sin hueco |
| 17 | 30 | 13,7 | 3/6 deriva al soltar, 1/6 `stack_collapse`, y **2/6 lo completan** |

Los números son de DESPUÉS de encender el filtro de alcance y el reintento de hueco: antes
eran 6,7 y 6,0, y la mitad de los fallos eran `ik_unreachable`. Ya no queda ninguno, así
que lo que miden ahora es lo que se quería medir —la decisión— y no la envolvente del
brazo.

**Se rompen por motivos distintos y sus números no se suman.** El 16 falla PLANIFICANDO:
veinte bultos del catálogo entero son el 85 % de la envolvente geométrica —los verdes se
quedan en 47-55 %— y colocar bien el bulto 18 exige haber colocado distinto el 3, que es
lo que un planificador incremental sin marcha atrás no puede hacer. El 17 falla
MANIPULANDO: de volumen va sobrado (43 %), pero la mitad de sus bultos son `pouch_xs` y
`mini_s`, los dos por debajo del suelo de huella de la celda, y una caja más pequeña que
la herramienta no baja junto a un vecino sin que el cuerpo del VGP20 lo toque.

Eso importa al decidir qué merece la pena aprender. El 16 es un problema de DECISIÓN y una
política tiene margen de sobra: reservar hueco para lo que aún no ha llegado es justo lo
que un greedy no hace. El 17 es en buena parte GEOMETRÍA DE LA HERRAMIENTA, y ahí una
política recupera parte —aprender a dejarle aire al vecino cuando el bulto es pequeño—
pero el resto no se arregla decidiendo mejor.

Y un detalle que se lee mal si no se avisa: un `ik_unreachable` en estos niveles **no es
ruido ajeno a la decisión**. Con `reach_max: 0` el planificador no tiene modelo de
alcance, así que elegir un hueco que el brazo no sostiene es una decisión suya, y por
tanto algo que se puede aprender a no hacer sin tocar la celda.

### El centro de gravedad

El paquete deja de tener el CoG en su centro geométrico: cada tipo lleva un
`cog_offset_m` en el catálogo. Eso toca **tres** sitios y hay que hacerlos los tres o la
mejora no significa nada:

1. **`vision/gauge.py` lo estima**, no lo lee del catálogo — es justo lo que la medición
   aporta. Si el gauge devuelve el valor exacto del YAML, sigues teniendo un oráculo con
   otro nombre. La implementación es una lectura de muñeca a plomo: las dos componentes
   horizontales salen exactas, la vertical se ancla al centro geométrico (nadie aguas
   abajo la lee) y `--precise-com` deja el barrido de cuatro poses cuando hace falta.
   Modelar el sello con una `weld` sesga el par; hay que reparentar el bulto a la
   herramienta. Ver `pesaje-en-el-sitio.md`.
2. **`planner/heuristic.py` lo puntúa**: un paquete con el CoG descentrado apoyado al
   borde del montón es un derrumbe con retraso. Es un término del score, no un filtro.
3. **`measure.pallet_state` lo usa** en vez del centro de la caja al acumular el CoG del
   palé. Esto es un cambio real sobre el `measure.py` portado, y es el único que hay que
   hacerle: mientras no se haga, la traza dice que el montón está centrado cuando no lo
   está.

`stability_margin` y `support_polygon` **se importan de `theker_telemetry`**, no se
reimplementan: una tercera copia acabaría discrepando con la que usa la interfaz, que es
el indicador que la define entera.

### La heurística

Mide la altura de todo el palé, y con las dimensiones del paquete que tiene en la mano
enumera todas las poses discretas que caben, descarta las inviables y puntúa el resto.
El score es un número entre 0 y 1 y el hueco elegido es el máximo.

Las cuentas están en **`placing/`**, en la raíz del repo: 9 ficheros, numpy y nada más.
`src/planner/heuristic.py` es sólo el adaptador. **Es el planificador por defecto.**
`--beam-planner` y `--naive-planner` existen para compararse contra él, y el modo que
se imprime al arrancar dice cuál está corriendo: dos ejecuciones que se comparan entre
sí tienen que distinguirse mirando la salida, no recordando qué banderas se pusieron.

**`placing/` no se toca.** Se copió entero y su frontera está blindada con un assert
ejecutable, no con una convención: `python -m placing` corre 15 comprobaciones y la
primera es que no se ha colado ningún módulo pesado en `sys.modules`. Esa frontera es lo
que permite ajustar los pesos sin arrancar MuJoCo —el ciclo pasa de minutos a
milisegundos— y se rompe con un solo import de conveniencia. Si algún día necesita un
dato del simulador, ese dato entra por el contrato o no entra.

Cuatro cosas que van en el diseño desde el principio:

- **El mapa de alturas se mide, no se lleva en un contador.** `heightmap.py` lo saca de
  las cámaras (`measure`) o de la escena (`measure_ground_truth`, el oráculo).
- **El score devuelve su desglose**, los 14 términos por separado. Va al `payload` del
  evento `plan` junto a `layer`, `slot` y `observed_pct` (sobrar es inocuo), y es lo que
  permite responder "¿por qué puso esa caja ahí?" sin volver a correr el episodio.
- **Un hueco sin candidato válido no es una excepción**: es un `wrong_placement`. El
  porqué viaja en `reject` del evento `fail`, y distingue "no cabe" —que cierra el palé—
  de "el brazo ya no llega" —que dice que el palé está mal puesto respecto al robot—.
  Y `{}` no es lo mismo que "todos rechazados": significa que no se generó ni un
  candidato, o sea que la caja no cabe ni en un palé vacío.
- **Para cambiar el comportamiento se tocan los pesos, no el código.** Los 14 están en
  `configs/pallet.yaml: heuristic.weights`; un peso a 0 apaga su término.

**Tres traducciones fallan en SILENCIO** y están las tres en la cabecera del adaptador:
`yaw` va en grados en `placing` y en radianes aquí; las alturas van en Z del mundo allí y
sobre la cubierta aquí —y hay que traducir el mapa **y** el límite de altura, porque medio
convenio funciona—; y `measure.Placement.position` viene en el frame del PALÉ, no del
mundo. Las tres tienen su test en `tests/test_pallet.py`, sin escena.

**`heuristic.max_stack_height` no es `pallet.max_height`.** 0.55 m sobre la cubierta es
hasta donde el barrido de IK de `tests/test_cell.py` valida el brazo; 1.35 es el papel
del palé. Y la cifra hace dos cosas: es el límite duro y es la ESCALA de `lowness`.
Ponerle el número grande diluye el gradiente 3.4x y la heurística se pone a hacer torre
en vez de llenar la capa. Medido.

**El filtro de alcance va apagado.** El adaptador no le pasa `robot_xy`, así que
`reachability` queda neutro y `out_of_reach` no descarta nada: las cifras de alcance que
trae `placing` son de un Panda. Mientras siga así, el planificador puede elegir un hueco
del palé al que el UR10e no llega, y eso sale como `ik_unreachable`. Encenderlo es medir
el barrido y poner tres números, no tocar código.

Ya existe una línea base medida en `tools/`: frente a first-fit, el beam search pasó de
50 % a 100 % de pilas estrictamente estables y redujo el descentramiento medio del CoM
de 142 mm a 81 mm. Cualquier heurística nueva se compara contra esos números.

### El ensayo de estabilidad

`src/cell/stability.py` somete la pila que dejó el robot a las **15 sacudidas de
transporte** (5 picos × 3 ejes, de 0.05 g a 0.80 g) y después a la **viga estrecha** en X
y en Y. Restaura la misma referencia antes de cada prueba, así que las 17 son
comparables entre sí. Las cifras son copia literal de `tools/scenarios/mixed_boxes.yaml`
y viven en `configs/pallet.yaml: stability_test`: se portan, no se recalibran a ojo.

Lo contrasta, no lo sustituye: `measure.py` ya da un margen de estabilidad **estático**
(`stability_margin` del SDK, contra el polígono de soporte). Esto es la prueba dinámica.

**El ensayo es opcional; la escena NO.** `--stability-test` decide si el ensayo corre,
pero tres cosas se compilan en **todos** los runs, con bandera o sin ella:

- el palé gana dos juntas de bisagra acotadas (`pallet_rx`, `pallet_ry`,
  `range="-0.0001 0.0001"`): pasa de 3 a **5 grados de libertad**;
- el cuerpo mocap `stability_beam` se monta siempre, aparcado en `z=-1` y con alfa 0;
- `rebuild()` conserva `qpos`/`qvel` del palé al recompilar, que es un cambio de
  comportamiento en el bucle de recoger y soltar, no en el ensayo.

**Consecuencia medida**, mismo nivel y semilla, sin la bandera, contra el `dev` anterior
(nivel 11, semilla 1): los resultados se desplazan **en el cuarto decimal** —`fill_ratio`
0.3163 → 0.3165, `lowness` +0.0001, `peak_penalty` −0.0001, `void_fill` −0.0001— y los
errores de colocación, **0.1 mm**. Frente a los 1.9 mm del `drop_clearance` calibrado es
ruido, y no degrada nada: 4/4 y `score` 1.0 en ambos. Pero está ahí: quien compare un run
de antes con uno de después verá los decimales moverse, y ésta es la línea que lo explica.

**Los resultados del ensayo NO son columnas.** No hay `kind` de evento para el ensayo ni
lo necesita: el resumen va dentro de `metrics` del episodio (≈0.5 KB, que es `jsonb` y
donde sobrar es inocuo) y el detalle de las 17 pruebas a `stability.json` en el
directorio del run, que no llega a la base.

### El entrenamiento de pesos sin robot

`src/training.py::simulate_placement` existe para ajustar los 14 pesos sin pagar el
coste ni mezclar los fallos del brazo. Construye la escena con `with_robot=False`, usa la
verdad del catálogo y el mapa oráculo, pone cada caja a `drop_clearance` sobre la pose de
`ScorePlanner` y deja que MuJoCo la asiente. Después ejecuta el MISMO ensayo de 15
sacudidas y dos vigas. No usa fuente, percepción, IK ni ventosa: sus resultados miden la
decisión y no sustituyen la validación del episodio entero.

La métrica `fallen_boxes_total` cuenta pares caja-ensayo con `fell_off=true`. Cada ensayo
restaura la pila de referencia, así que si el mismo paquete cae en dos pruebas cuenta dos
veces; `unique_fallen_boxes` conserva además el número de ids distintos. La función de
pérdida es `missing * (n_trials + 1) + fallen_boxes_total`: una política que no coloca
nada no puede ganar artificialmente con cero caídas.

`scripts/train_weights.py` aplica CEM, una búsqueda de política episódica sin dependencias
nuevas. **No escribe los pesos aprendidos en `configs/pallet.yaml`.** Deja la propuesta en
`best_weights.json`; promoverla a configuración exige repetir la escalera normal con brazo
y documentar la medida junto a los pesos.

Los logs son parte del contrato del entrenador. Cada run crea
`runs/<timestamp>-weight-training/` con `manifest.json`, `config_snapshot.json` y su
SHA-256, `source_snapshot/` con el código exacto aunque el worktree esté sucio,
`progress.jsonl` sincronizado a disco después de cada colocación, ensayo y evaluación,
el detalle completo bajo `evaluations/`, `best_weights.json` y `summary.json`. No se
sustituye ese JSONL por una barra de progreso: si el proceso se corta, la última fila
tiene que decir exactamente qué evaluación estaba corriendo.

### El decorado

`levels[].decor` dice en qué sitio pasa el nivel. Es un vocabulario cerrado de dos
—`cell | plant`— y arrastra el DECORADO y LA LUZ a la vez, porque una nave con las luces
de un plató no es una nave.

- **`cell`** es lo de siempre, y el valor por defecto: valla de seguridad, marcas de
  suelo, suelo de damero y las tres luces de `lighting:`. Los once niveles que ya había
  no se enteran de que esto existe.
- **`plant`** monta la planta industrial de `src/cell/plant.py` alrededor de la celda:
  paredes, ventanas, luminarias, estantería, depósitos, banco y señalización. Hoy sólo
  lo usa el nivel 34.

Cuatro cosas que conviene tener claras antes de tocarlo:

1. **El decorado es un FONDO y no colisiona.** Ni un geom de la nave tiene contacto, y
   ése es el motivo de que el 33 y el 34 salgan idénticos hasta el último decimal con la
   misma semilla. Si algún día algo de la nave tiene que ser sólido, deja de ser gratis:
   hay que volver a barrer el alcance y volver a medir los dos episodios.
2. **La nave viene de fuera.** `assets/planta_industrial_v2.xml` es una COPIA de una
   exportación de Blender que vive en `~/Descargas/planta_industrial/`, y allí
   `export_mujoco.py` la sobrescribe cada vez que alguien re-exporta. La del repo es la
   que manda. `plant.py` se queda sólo con su `<worldbody>` y tira sus luces, su cámara,
   su suelo y sus tres cuerpos libres — los cuatro chocan con esta celda, y los cuatro
   fallarían tarde. El porqué de cada uno está en la cabecera del módulo.
3. **Cada decorado trae sus propias medias medidas.** No hay una tabla de saturación,
   hay dos: la de `lighting:` y la de `plant.lighting:`, cada una con su barrido. Cambiar
   la luz de un decorado NO invalida la del otro, y ésa es la razón de separarlas en vez
   de parametrizar una sola.
4. **Las direccionales de la nave no proyectan sombra.** El mapa de sombras es uno para
   toda la escena, y con la nave alrededor la extensión pasa de 4 m a 14: una direccional
   se queda con menos de una décima parte de los téxeles y sale escalonada por las
   paredes. La sombra la hacen las tres luminarias, que son locales.

## 7. Lo que se porta, y no se reescribe

El robot ya está decidido: **UR10e + OnRobot VGP20**. La ejecución procede del
demostrador `stable_pallet`, que queda completo en `tools/`; las funciones de fila de
telemetría siguen procediendo de `STACKSPECT/Guionized-simulation` porque ya estaban
verificadas contra el esquema real.

| Origen | Destino | Qué se conserva |
|---|---|---|
| `tools/stable_pallet/simulator.py` | `src/cell/scene.py`, `arm.py`, `render.py` | MJCF, UR10e, IK amortiguada, corrección cartesiana, VGP20 y cámaras |
| `tools/stable_pallet/simulator.py`, `shake.py` | `src/cell/stability.py` | pila post-paletizado, 15 sacudidas restauradas y viga estrecha X/Y |
| `tools/stable_pallet/truck.py` | `src/cell/truck.py` | carga, orden de descarga y deriva |
| `tools/stable_pallet/planner.py` | `src/planner/naive.py::BeamPlanner` | beam search, lookahead y modelo de estabilidad interno |
| `Guionized-simulation/src/pallet/measure.py` | `src/measure.py` | geometría y `demo()`; el CoG añade `cog_offset_m` |
| `Guionized-simulation/src/pallet/telemetry.py` | `src/telemetry.py` | claves de fila y ciclo de vida en vivo |

Las calibraciones están en las cabeceras de `configs/scene.yaml` y
`configs/pallet.yaml`, con la medida al lado:

- `reach_tolerance`: 20 mm, por encima del suelo medido del servo.
- `cup_gap`: 1,5 mm entre almohadilla y cartón al sellar.
- `clearance`: 40 mm entre cajas de una capa; con 20 mm había roces.
- `drop_clearance`: 2 mm; la ventosa no necesita abrir dedos.
- generador: 18–55 × 16–40 × 8–30 cm, 90–280 kg/m³ y masa máxima 8,5 kg.
- bahía alcanzable: 0,74 × 0,88 m en `scene.yaml`; el barrido documentado dio 0 de 720
  poses fuera de alcance. Si cambian robot, herramienta o bahía, se repite el barrido.

## 8. Qué NO hacer

- **No metas conocimiento de la plataforma fuera de `src/telemetry.py`.** Es la única
  frontera: el resto del código no debería conocer el nombre de ninguna columna. Lo que
  cruza los módulos son objetos de `contracts.py`, no filas.
- **No reimplementes `stability_margin` ni `support_polygon`.** Vienen del SDK.
- **No inventes `kind`s de evento, vistas de foto, causas de fallo ni decorados.** Los
  cuatro son vocabularios cerrados y los cuatro fallan tarde y en silencio.
- **No des contacto al decorado.** La nave de `src/cell/plant.py` es un fondo: si empieza
  a chocar, deja de ser gratis y el codo se engancha en un depósito. Y no re-exportes su
  XML encima del de `assets/`: el de `~/Descargas` lo pisa Blender, el del repo manda.
- **No hagas que un módulo lea la escena por su cuenta.** Si `heuristic.py` importa
  `mujoco`, algo se ha torcido: lo que necesita es el `Heightmap` y el `PackageSpec`. La
  excepción son los stubs-oráculo, que existen precisamente para hacer trampa, y por eso
  viven en ficheros aparte con el nombre puesto.
- **No toques `configs/` a ojo.** Cada valor raro tiene su medida al lado; si cambias uno,
  deja escrito cómo lo mediste.
- **No añadas dependencias** sin mirar antes si MuJoCo o numpy ya lo hacen.
- **No metas nada dentro de `placing/`**, ni un import de conveniencia. Es código copiado
  de otro repo y su valor es justamente que no sabe que existe un simulador. Si hace
  falta un dato de la escena, entra por el contrato o no entra; y lo que haya que
  adaptar se adapta en `src/planner/heuristic.py`, que para eso está.

## 9. Cómo se comprueba, en este orden

0. **El módulo de la heurística, solo:** `python -m placing`. Quince comprobaciones sin
   simulador ni red, y la primera es la frontera de imports, que es lo que se rompe al
   integrar. Cuesta un segundo: que se quede en el CI.
1. **Sin red ni simulador:** `python tests/test_pallet.py`. Que las claves de cada fila
   sean las columnas de su tabla, que los `seq` no se repitan, que las vistas y las
   causas de fallo estén en su vocabulario, y las tres traducciones del adaptador —yaw en
   radianes, la altura 0 en la cubierta y no en el suelo, y el frame del palé en
   `record`—, que son las que fallan sin dar error. Falla en segundos, no tras tres
   minutos de simulación.
2. **La medida, sola:** `python -m src.measure`. Sus asserts cubren el CoG con cajas
   fuera de tolerancia y el margen contra el polígono de soporte.
3. **La celda:** `python tests/test_cell.py`. Compila las tres fuentes, comprueba las
   cámaras, la banda, el orden del camión, la envolvente de IK, el pesaje de muñeca y
   que el ensayo físico deja la misma pila que encontró.
4. **Con visor:** `python scripts/palletize.py --viewer --source table`.
5. **Sin Supabase:** `python scripts/palletize.py -n 1 --no-telemetry --level 21`. Un episodio
   entero a disco; mira el `episodes.jsonl`.
6. **Con telemetría, y que diga en qué modo va.** Si no imprime que está activa, no lo
   está.
7. **Con `/` abierto en el navegador.** Es la prueba de verdad: el episodio aparece **en
   curso** a los pocos segundos, el palé se monta paquete a paquete, los KPIs se mueven
   solos y al acabar pasa a terminado con su éxito o su causa de fallo.
8. **El demostrador:** `cd tools && uv run --extra dev pytest`. El extra es explícito:
   con un entorno nuevo, `uv run pytest` puede encontrar el ejecutable global antes de
   instalar las dependencias de desarrollo y coleccionar con el Python equivocado.
9. **El panel:** un nivel de mesa, cinta y camión en EJECUCIÓN y DEPURACIÓN; el modo
   debe estar visible y DEPURACIÓN debe avisar que no publica. Cambiar de nivel tiene
   que cambiar la carga (catálogo, número, ruido, CoG), no solo el título de la
   tarjeta: DEPURACIÓN no puede colapsar las nueve en el experimento legado. Y ningún
   control del panel sin bandera en `scripts/palletize.py` se queda encendido: se
   desactiva y se escribe al lado por qué. Un control inerte cuesta más de depurar que
   uno que no está. **Ensayo de estabilidad al terminar** sí tiene bandera y va
   encendida: repite un nivel con ella y deben aparecer las 15 sacudidas y la viga X/Y.
10. **Contra la base**, después:

```sql
-- una fila de traza por paquete intentado
select count(*) from pallet_states where episode_id = '<id>';
-- vacío: ningún seq repetido
select seq from events where episode_id = '<id>' group by seq having count(*) > 1;
-- las fotos, y que su after_seq case con la traza
select after_seq, view, width, height from snapshots
 where episode_id = '<id>' order by after_seq, view;
-- basura de ejecuciones abortadas: tiene que ser 0
select count(*) from episodes where status = 'running';
```

Si la interfaz no reacciona, descarta esto antes de buscar en tu código: nada nunca → no
estás subiendo; el palé aparece ya montado → estás usando `episode()` en vez de
`begin()`/`end()`; Live clavada en un episodio viejo → un huérfano en `running`; las
cajas diminutas en un palé enorme → falta `config.pallet_size_m`; la ejecución no sale en
Ejecuciones → esa pantalla no se refresca sola, recarga.
