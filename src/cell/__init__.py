"""
La celda: el banco de trabajo. Mesa, frame del TCP y orientación de cámaras.

PORTAR TAL CUAL desde `~/HackSpain/paletizado-guionizado/src/cell.py`
(remoto: `STACKSPECT/Guionized-simulation`).

Lo que trae y se usa aquí sin cambios:

    TCP_SITE                       nombre del site que mink usa como frame del TCP
    add_table(spec, scene_cfg)     la mesa, a partir de configs/scene.yaml
    tcp_frame(position, closing)   pose de agarre desde un punto y una dirección de cierre
    lookat_quat(position, lookat)  cuaternión de una cámara que mira a un punto

No lo reescribas "mejor": está medido contra el alcance real del brazo y `tcp_frame` es
lo que hace que todas las poses compartan orientación, que es de donde sale que el
trayecto en línea recta funcione.
"""
