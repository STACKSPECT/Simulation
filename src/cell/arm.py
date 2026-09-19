"""
El control del brazo: IK cartesiana con mink, pinza y bucle de física.

PORTAR TAL CUAL desde `~/HackSpain/paletizado-guionizado/src/control/arm.py`.

Expone `ArmController` con:

    tcp_pose()                 pose actual del TCP
    move_to(pose)              waypoint cartesiano; False si no converge -> ik_unreachable
    move_joints(qpos, dur)     espacio de juntas; SOLO para la foto final
    set_gripper(value, hold_s) 0-255 sobre el recorrido completo
    is_holding(width)          False si la caja se escurrió -> grasp_slip
    step_physics()             un tick de control; es lo que refresca el visor
    steps_per_tick

Tres cosas que ya están resueltas ahí y que cuesta caro volver a descubrir:

  - **Las tolerancias de convergencia van POR ENCIMA del suelo medido del servo** (~4 mm
    de caída estacionaria bajo carga). Por debajo, `move_to` no converge nunca y todo
    acaba declarado `ik_unreachable`.
  - **Todo el trayecto va en cartesiano a una altura de tránsito fija**: subir recto,
    cruzar, bajar recto. El rodeo por `home` en espacio de juntas tira la caja al ir y
    barre el montón al volver.
  - **`posture_cost` ancla los DOF sobrantes** sin pelearse con la tarea del TCP.

Los valores de `motion` viven en `configs/scene.yaml` y los pisa `configs/pallet.yaml`
para esta celda. Son knobs de calibración del hardware simulado: si cambia el robot —y
está sin decidir— se vuelven a medir, no se copian.
"""
