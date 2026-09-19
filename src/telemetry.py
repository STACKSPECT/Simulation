"""
La costura con la plataforma: medidas del episodio -> filas de Supabase, en vivo.

PORTAR desde `~/HackSpain/paletizado-guionizado/src/pallet/telemetry.py`. Sus funciones
de fila están verificadas contra el esquema real y **los nombres de sus claves SON el
contrato**: cada una es una columna.

**Este es el único módulo de la simulación que sabe que la plataforma existe.** Todo lo
demás produce medidas y no tiene ni idea de dónde acaban ni de cómo se llaman las
columnas. Si mañana cambia el contrato, se toca aquí y en ningún otro sitio.

Se porta:

    run_config(scene)                  lo que va a `runs.config`
    RunLogSink                         lo que `run_episode(..., sink=)` va llamando
    placement_row(index, placement)    una caja intentada
    pallet_state_row(index, st, drift) ESTO es la traza de centro de gravedad
    episode_result(episode, scene)     el resumen, con `metrics`
    save_snapshots(episode, directory) las fotos a disco, pase lo que pase

Las reglas, que están todas en AGENTS.md §5 y ninguna se ve leyendo el SDK:

  - Ciclo de vida `begin() -> event/placement/pallet_state/snapshot -> end()`, nunca
    mezclado con `episode()`. Es lo único que hace que la pantalla Live esté viva.
  - Columnas en SI; `payload` de eventos en mm y grados. En el payload sobrar es inocuo
    y faltar rompe en silencio; en el nivel de la fila, sobrar es un 400 que apaga la
    subida del resto de la ejecución.
  - Una fila de `pallet_states` por paquete INTENTADO, incluido el que derrumba el
    montón: es la que hace que la traza cruce el cero.
  - El disco primero. Los fallos de red no se gestionan aquí: el SDK avisa una vez,
    apaga la subida y el episodio sigue. **No envuelvas esto en try/except.**
  - `snapshot()` entre `begin()` y `end()`, y `view` en `top|side|iso|camera`.

Lo que CAMBIA respecto al guionizado:

  - **`oracle`**. Allí era `True` fijo, porque las poses se conocían. Aquí depende de
    qué piezas se estén usando: es `any(stub en uso)` y lo calcula `scripts/palletize.py`,
    que es quien las elige. Marcarlo mal invalida justo la comparación que justifica el
    trabajo, porque la interfaz no compara un run con oráculo contra uno sin él.
  - **`metrics`** gana lo que aporta el planificador: el score medio de las colocaciones
    y, si se quiere, el desglose agregado. `score` sigue siendo el del episodio
    (colocadas / intentadas), que es lo que lee la vista.
  - **`run_config`** describe además la cinta y el catálogo de paquetes. Lo que no puede
    faltar es `pallet_size_m`: sin él la interfaz supone un europeo de 1200x800 y, con
    una maqueta, todas las cotas salen mal por el factor de escala.
  - **`failure`** puede ser `no_detection`, que allí era inalcanzable.
"""
