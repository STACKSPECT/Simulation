"""Fuente de mesa: una cinta saca los bultos de la caja negra y los deja en la mesa.

── POR QUÉ UNO, Y NO LA MESA LLENA ──────────────────────────────────────────────

Estaba llena, repartida en rejilla entre el NÚMERO de bultos y sin mirar cuánto medían.
Eso daba tres cosas, y ninguna se veía como lo que era:

  - **Bultos solapados.** Nivel 12, seis tipos: 240 mm de separación para cajas de 340 mm
    de fondo. Seguían deslizándose con el episodio ya empezado —0.019 m/s tras el
    asentado— así que la caja llegaba a la ventosa 23.8 mm corrida y el episodio moría en
    `wrong_placement` por 2 mm de tolerancia. Parecía que se escurría de la ventosa.
  - **Bultos colgando del canto.** Nivel 13, ocho sorteados de hasta 550 mm: velocidad
    residual 3.89 m/s, que es caída libre.
  - **Y el vecino por los aires.** Éste sólo sale con movimiento real; en fast-forward el
    brazo teletransporta y no se ve. Al extraer un bulto, el que lleva en la mano barre
    al de al lado y lo tira al suelo.

Esto último es lo que decide el diseño, porque **no tiene arreglo con más holgura**. La
mesa mide 0.72 x 0.68 m útiles. Dos `std_m` (0.42 x 0.30) sólo entran de canto en Y, y
ahí la holgura máxima que cabe son 80 mm:

    holgura   vecina tras extraer      cabe en la mesa
     30 mm    384 mm, al suelo         sí
     80 mm    385 mm, al suelo         sí (al límite)
    150 mm    —                        NO, se sale del canto

Es decir: la holgura que hace falta para extraer sin tocar es mayor que la que cabe. Con
esta mesa y esta herramienta, la única configuración que no derriba nada es **un bulto**.

── DE DÓNDE SALE ESE BULTO ──────────────────────────────────────────────────────

Antes aparecía ya puesto en el centro de la mesa, y los demás esperaban aparcados en fila
fuera de la escena, en `x = -3 - i*0.7`: con treinta bultos son 21 m de cartones
perdiéndose en el horizonte. Ahora **llegan por una cinta que sale de la caja negra**, que
es lo que pasa en una planta y además cuenta la verdad: el sistema no sabe lo que viene.

Y por eso esta clase es un `Belt` y no otra cosa. La mecánica es la misma y lo único que
cambia es **dónde acaba el carril**: `scene.lane_for` da el de la cinta a los niveles 2x,
que entrega bajo el brazo, y el de la mesa a los 1x, que entrega a la mesa. La banda de
mesa acaba justo en su canto y comparte cota con ella —las dos a 0.58—, así que el cartón
cruza sin escalón; y como `Belt._riding` mira la ALTURA y no la x, la banda lo sigue
arrastrando ya sobre la mesa y lo para en su centro, que es donde más canto le queda por
los cuatro lados. No hay teletransporte en ningún punto del trayecto.

Lo que esto NO cambia es el resto del contrato: sigue habiendo un bulto cada vez, el
siguiente entra cuando el brazo se ha llevado el anterior, y un carril que no entrega
antes de `timeout_s` sigue siendo un `timeout` y no una causa de fallo inventada.
"""

from __future__ import annotations

from src.cell.conveyor import Belt


class TableSupply(Belt):
    """La cinta de la caja negra, que acaba en la mesa en vez de bajo el brazo.

    Sin cuerpo propio a propósito: cualquier cosa que hubiera aquí sería mecánica de
    cinta duplicada, y dos copias de eso acaban discrepando. Lo único que distingue a
    esta fuente vive en `scene.lane_for`, que es un dato, no código.
    """
