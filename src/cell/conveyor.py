"""
La cinta: entrega paquetes a la estación de recogida y se para para que el brazo coja.

SIN IMPLEMENTAR. No existe en el repo guionizado —allí las cajas esperaban colocadas en
la mesa— así que esto se escribe de cero.

El contrato es corto:

    present(scene) -> str | None   avanza hasta que un paquete llega a la estación y
                                   PARA. Devuelve su id, o None si no llegó nada.
    release(scene)                 vuelve a arrancar, cuando la mano ya no está encima.

Y tiene dos avisos que no se ven leyendo el código:

**La cinta no tiene evento propio.** `events.kind` es un CHECK cerrado de seis valores
—`perceive | plan | pick | place | settle | fail`— y ninguno es suyo. Su estado viaja en
el `payload` del evento `perceive`, donde sobrar es inocuo, o no viaja. Inventar un kind
hace que la base rechace la fila con un 400 que el SDK se traga, y a partir de ahí se
apaga la subida del resto de la ejecución.

**Un atasco es `timeout`, no una causa nueva.** No hay valor en el vocabulario para "la
cinta no entregó". Si la estación se queda vacía y expira la espera, es `timeout`; si
entrega pero la percepción no ve nada, es `no_detection`. Son cosas distintas y el
gráfico de fallos las separa, así que no las mezcles.

Cómo moverla es cosa tuya, pero lo barato es un `slide` con actuador de velocidad, o
aplicar velocidad a los cuerpos en contacto con el geom de la banda: parar es poner la
consigna a cero. Lo que NO vale es teletransportar el paquete a la estación — el punto
de tener cinta es que el paquete llegue con la pose que traiga, incluida la torcida.

**`release()` va después de que el brazo se haya retirado**, no antes: arrancar con la
mano dentro arrastra el siguiente paquete contra la pinza.
"""

from __future__ import annotations


def present(scene) -> str | None:
    """Avanza la cinta hasta la estación de recogida y para. `None` = no llegó nada."""
    raise NotImplementedError("cinta sin implementar")


def release(scene) -> None:
    """Vuelve a arrancar la cinta. Solo con el brazo ya retirado."""
    raise NotImplementedError("cinta sin implementar")
