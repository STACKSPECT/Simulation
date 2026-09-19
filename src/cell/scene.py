"""
La escena: brazo, mesa, palé, cinta, paquetes, cámaras y luces.

PORTAR desde `~/HackSpain/paletizado-guionizado/src/pallet/scene.py`, con cambios.

**Se construye con `mujoco.MjSpec` sobre el `scene.xml` del Menagerie, no con XML
propio.** Así no hay que pelearse con `meshdir` ni con rutas relativas de `<include>`, y
cambiar de robot es cambiar una ruta en `configs/scene.yaml`. Mantén eso.

Se porta tal cual:

    PalletScene, Box, load_configs, build_scene, box_pose
    _add_pallet, _set_lighting, _camera_quat

La iluminación en particular: está calibrada con medias RGB medidas para que la cenital
no sature y el alzado no salga a oscuras. Los números y el porqué, en la cabecera de
`configs/pallet.yaml`.

Lo que CAMBIA respecto al guionizado:

  - **Las cajas ya no salen de un guion.** Allí `build_scene` recorría `script` y dejaba
    cada caja en su sitio de preparación. Aquí el catálogo genera los paquetes y la
    cinta los va entregando: la escena ya no sabe dónde va a acabar ninguno, y eso es
    justo lo que la diferencia.
  - **`layer_heights`, `layer_base_z` y `planned_pose` se van.** Eran la geometría del
    guion —dónde DEBE quedar cada caja, calculado de antemano— y ahora eso lo decide el
    planificador. Lo que las sustituye es `PlacementPlan.position`.
  - **Entra la cinta** (ver `cell/conveyor.py`) y con ella una estación de recogida.
  - **Cada tipo de paquete lleva `cog_offset`**: el CoG deja de estar en el centro
    geométrico. En MuJoCo eso es la `pos` del geom dentro de su body, o un `inertial`
    explícito; decídelo en un sitio y déjalo escrito.

`build_scene` sigue devolviendo la escena con la física ya asentada. Medir antes de eso
es medir cajas todavía cayendo.
"""
