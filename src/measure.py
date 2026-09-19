"""
La medida: dónde acabó cada paquete, cuánto apoya, cuánto sobresale, y cómo está el
montón. De aquí sale todo lo que se sube a la plataforma.

PORTAR desde `~/HackSpain/paletizado-guionizado/src/pallet/measure.py`, con UN cambio.

Se porta entero, incluido su `demo()` de asserts —`python -m src.measure` es el segundo
paso de la comprobación— y su API:

    Placement, PalletState                 lo medido de un paquete y del montón
    footprint, rect_overlap                geometría de huellas
    support_ratio, overhang                apoyo y vuelo
    to_pallet_frame, pallet_rect           mundo -> frame del palé (el que dibuja la UI)
    on_pallet(scene, placements)           las que TOCAN el palé, criterio geométrico
    measure_placement(scene, box, i, done)
    pallet_state(placements)               masa, CoG, margen, relleno
    fill_ratio, layer_flatness
    positions, max_displacement            la deriva al asentarse

Tres decisiones de ese fichero que hay que conservar literalmente:

  - **`pallet_state` cuenta las cajas fuera de tolerancia.** Una caja mal puesta sigue
    pesando y sigue moviendo el CoG. Filtrar por `placed` hace que la traza acabe en
    verde en un episodio que se cayó. Allí lo cubre
    `test_the_cog_counts_boxes_outside_tolerance`; tráete el test también.
  - **Lo que NO cuenta es lo que se quedó en la mesa**, y el criterio es geométrico:
    ¿su huella toca el palé? No "¿falló la maniobra?". Ver `on_pallet`.
  - **`footprint` sobreestima a propósito** (envolvente, no polígono rotado), para que
    los errores sesguen pesimista.

Y `stability_margin`, `support_polygon` y `TRANSPORT_ACCEL_G` **se importan de
`theker_telemetry`**: es el indicador que define la interfaz entera, y una tercera copia
acabaría discrepando. El margen se mide contra el polígono de soporte —la envolvente de
las huellas de la capa 1—, no contra el borde del palé: contra el palé los números salen
optimistas y la pantalla dice que todo va bien hasta el derrumbe.

── EL CAMBIO ────────────────────────────────────────────────────────────────────

**`pallet_state` acumula el CoG con `cog_offset_m`, no con el centro de la caja.** Allí
el centro de gravedad de cada paquete era su centro geométrico porque lo era; aquí no, y
mientras no se haga este cambio la traza dice que el montón está centrado cuando no lo
está. Es el único sitio de la medida que toca el CoG de los paquetes, y va con su
assert en `demo()`.
"""
