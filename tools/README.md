# `tools/` — el demostrador de paletizado estable

Herramienta auxiliar para pruebas rápidas. **No es el punto de entrada del repo**: el
entrypoint es `scripts/palletize.py` y la razón está en la cabecera de
`stable_pallet/cli.py`. Nada de aquí abre episodios ni sube filas a la plataforma.

Esto es el proyecto del que salió la capa de ejecución de `src/cell/`, traído entero
para que no se pierda lo que no cabía en las fronteras del repo: el planificador con
*beam search*, el modelo de estabilidad, la báscula de muñeca, el ensayo de transporte,
el benchmark y el panel de control.

## Qué contiene

| Módulo | Qué hace |
|---|---|
| `planner.py`, `stability.py`, `geometry.py` | Beam search con *lookahead*, polígono de soporte, propagación descendente de cargas |
| `com_probe.py`, `com_estimator.py`, `tare.py`, `wrench.py` | Báscula de muñeca: una lectura a plomo por defecto; `--precise-com` barre varias poses |
| `truck.py` | Cómo llega cargado un remolque y el único orden en que puede vaciarse |
| `suction.py` | OnRobot VGP20: qué bloque de ventosas cabe en cada caja y con qué capacidad |
| `shake.py` | 15 sacudidas de transporte (0,05–0,80 g en X, Y, Z) y ensayo de viga estrecha |
| `benchmark.py` | Comparación determinista contra un *first-fit* |
| `generator.py` | Sobre de cartón: 18-55 × 16-40 × 8-30 cm, 90-280 kg/m³, masa = densidad × volumen |
| `webapp.py`, `runner.py`, `playback.py` | Panel: ambos modos lanzan `palletize.py`; depuración con `--no-telemetry` |
| `simulator.py` | La celda original, de la que se troceó `src/cell/` |

## Cómo se usa

```bash
cd tools
uv sync --extra dev --extra video

uv run stable-pallet plan --scenario scenarios/mixed_boxes.yaml    # sin física
uv run stable-pallet simulate --viewer --no-measure-com            # ver la celda
uv run stable-pallet shake --instant-place --no-measure-com        # solo el palé
uv run stable-pallet benchmark --trials 20                         # contra first-fit
uv run pytest                                                      # 200 comprobaciones
```

El panel muestra trece tarjetas, tomadas de `configs/pallet.yaml`, y un selector de
modo. **EJECUCIÓN** llama a `scripts/palletize.py` con la fuente y el nivel de la
tarjeta y puede publicar telemetría. **DEPURACIÓN** llama al mismo entrypoint con
`--source`, `--level` y `--no-telemetry`: misma escena, sin subir nada.

Cada ajuste viaja como bandera al arrancar; el hijo informa por una tubería de ida y no
escucha. Lo que no tiene bandera sale apagado del panel, con su porqué escrito al lado:
la barra de reproducción (`--protocol json` no emite `state`, así que no hay grabación),
«Pesar cada caja en la muñeca» (issue #25) y «Dejar el visor abierto al acabar». Las dos
casillas de centro de masa comparten `--show-com`, y «Ensayo de estabilidad al terminar»
manda `--stability-test` en los dos modos: sí tiene bandera, va encendida, y el título
del run acaba en `· ESTABILIDAD` para que un run con ensayo se distinga de uno sin él.

El aspecto del panel es el de la marca STACKSPECT: `stable_pallet/web/tokens.css` es una
copia de `Platform/frontend/styles/colors.css` y `app/globals.css` (colores en claro y
oscuro, radios, tipografías), y `web/brand/`, `web/fonts/` y los iconos salen del mismo
sitio. Las fuentes (Plus Jakarta Sans y Geist Mono) van autoalojadas para que el panel no
pida nada a la red. Para re-sincronizarlo, copia los valores de esos ficheros a
`tokens.css`; `app.css` y `app.js` no llevan ningún color literal, solo `var(--…)`.

Los escenarios se buscan primero desde donde estés y después desde `tools/`, así que
`scenarios/mixed_boxes.yaml` funciona igual desde la raíz del repo que desde aquí.

## Modelo de estabilidad

Para una caja, el sistema intersecta su huella con todas las superficies cuya cota
superior coincide con su base; las esquinas de esas áreas forman el polígono convexo de
soporte. La configuración solo es válida cuando la superficie apoyada supera el umbral,
la resultante de la masa propia y las cargas superiores cae dentro del polígono con
margen, y esa carga se transmite hacia abajo en proporción al área de contacto sin
romper ninguna interfaz inferior.

Es una aproximación cuasiestática conservadora y rápida, adecuada para evaluar miles de
candidatos; MuJoCo actúa como segunda validación dinámica al soltar. No sustituye a un
solver completo de equilibrio con fricción.

> **Ojo:** este indicador es del planificador, no el que se publica. El margen de
> estabilidad que ve la interfaz sale de `stability_margin` de `theker_telemetry`, y la
> regla del repo es que no haya una segunda copia. Ver `src/measure.py`.

## Qué mide

Del planificador: tasa de finalización, estabilidad estricta y descentramiento del CoM.
Con la semilla 17 y 10 permutaciones del escenario principal ambos métodos colocan el
100 % de las cajas, pero el *first-fit* solo produce una pila estrictamente estable en
el 50 % de los ensayos frente al 100 % del beam search, y el descentramiento medio del
CoM baja de 142 mm a 81 mm.

De la descarga: el orden de extracción real, si coincide con el único orden legal
—siempre la caja más alta— y cuánto se movió el resto de la carga mientras esperaba su
turno. Con la semilla 0 esa deriva máxima es de 0,4 mm en las seis cajas.

## Limitaciones honestas

- La bahía del remolque (0,74 × 0,88 m) es el trozo al que llega un brazo fijo, no un
  semirremolque entero. Aquí se modela el problema de decisión, no el de desplazamiento.
- El brazo colisiona con el remolque pero no con los paquetes: las holguras contra la
  carrocería se han medido, no se calculan en línea.
- Las cajas son paralelepípedos rígidos; no se modela deformación de cartón.
- El vacío se aproxima con una restricción soldada mientras el agarre está activo. La
  capacidad sí sale de las ventosas cubiertas, pero no se simulan fugas ni porosidad.
- La sacudida no modela la suspensión de un camión ni un perfil ASTM D4169 completo.

El modelo del UR10e deriva de MuJoCo Menagerie; la atribución BSD-3-Clause está en
`THIRD_PARTY_NOTICES.md`, en la raíz del repo.
