"""
Las comprobaciones que corren SIN simulador y SIN red, en segundos.

PORTAR desde `~/HackSpain/paletizado-guionizado/tests/test_pallet.py`. Son asserts
planos, sin framework: `python tests/test_pallet.py`.

Existen porque el resto de la comprobación es cara: un episodio son minutos de física, y
descubrir ahí que una clave está mal escrita significa haber tirado la ejecución entera.
Peor: media docena de los errores que este contrato permite **no dan error**. Una clave
de payload que falta deja la interfaz en blanco sin avisar; una columna mal escrita es un
400 que el SDK se traga y que apaga la subida del resto del run. Esto los caza antes.

Lo que se porta tal cual:

  - que las claves de cada fila sean exactamente las columnas de su tabla
  - que los `seq` de los eventos no se repitan dentro de un episodio
  - que las vistas de las cámaras estén en `top|side|iso|camera`
  - que las causas de fallo estén en el vocabulario de ocho
  - `test_the_cog_counts_boxes_outside_tolerance` — una caja mal puesta sigue pesando
  - los cuatro casos que anclan `stability_margin` contra el polígono de soporte

Lo que hay que AÑADIR aquí, que allí no tenía sentido:

  - **`events.kind` está dentro de los seis permitidos.** Con cinta de por medio la
    tentación de inventar un kind es real, y la base lo rechaza tarde.
  - **La heurística devuelve un score en [0, 1] y un `breakdown` no vacío.** Sin esto un
    planificador que siempre devuelve 0.0 pasa desapercibido.
  - **La heurística devuelve `None` cuando no cabe nada**, en vez de lanzar.
  - **Los pesos de `configs/pallet.yaml` suman 1.0.** Es la clase de error que solo se
    nota como "el planificador se ha vuelto raro".
  - **`oracle` es `any(stub en uso)`** para cada combinación de banderas.
"""
