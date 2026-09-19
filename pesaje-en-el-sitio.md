# Medir el centro de masas de un bulto con una sola lectura de muñeca

Guía de aplicación del método que usa esta celda. Sirve para cualquier sistema que agarre
un objeto con una herramienta instrumentada y necesite saber dónde tiene la masa, no solo
cuánta.

La idea, en una frase: **la componente del centro de masas que una sola pose no puede ver
es la paralela a la gravedad, y si agarras por arriba esa es la altura del CoM dentro del
bulto — que casi ningún planificador de paletizado lee.** Así que no la midas: supónla
centrada, mide las otras dos exactas, y comprueba que la dirección que has supuesto sigue
siendo la que no importa.

## 1. Cuándo aplica

Antes de copiar nada, estas cinco condiciones. Si falla la tercera o la cuarta, este método
no es para tu caso.

1. **Sensor de fuerza/par de seis ejes** entre la herramienta y lo que sostiene, o algo
   equivalente (celdas de carga en la brida, corrientes de los ejes de la muñeca).
2. **El objeto se queda quieto respecto a la herramienta.** Ventosa sellada, garra cerrada,
   imán. Si se balancea o se escurre, esto mide el balanceo.
3. **La herramienta trabaja en una orientación conocida y repetible**, típicamente a plomo.
   Es lo que hace que la dirección ciega sea siempre la misma.
4. **Lo que hay aguas abajo no usa la componente vertical del CoM.** Compruébalo en el
   código, no de memoria. En nuestro caso: el chequeo de estabilidad acumula momentos en X e
   Y, el margen al vuelco es una distancia en el plano de contacto y la función de coste
   mira el CoM global en planta. La altura del CoM dentro del cartón no entra en ninguna
   decisión de colocación.
5. **Sabes dónde está el centro geométrico del objeto respecto al sensor.** Si tú eliges el
   agarre, lo sabes exacto. Si te lo da una cámara, lo sabes con el error de la cámara.

## 2. La física

Con el objeto quieto, el sensor ve el peso de todo lo que cuelga de él:

```
F = -m · g_s
τ = r × F = -[F]_× · r
```

`g_s` es la gravedad expresada en el frame del sensor, y sale de la cinemática directa (la
orientación de la brida), no de ninguna medida extra. `r` es el vector del origen del sensor
al centro de masas. La segunda ecuación es lineal en `r`, que es lo que hace esto un despeje
y no un ajuste.

Con la herramienta a plomo, `g_s = (0, 0, -g)` y por tanto `F = (0, 0, m·g)`:

```
τ_x =  r_y · m·g      →   r_y =  τ_x / (m·g)
τ_y = -r_x · m·g      →   r_x = -τ_y / (m·g)
τ_z =  0              →   r_z no aparece
```

Las dos componentes horizontales son una división. **No hay aproximación en ellas.** La
tercera es estructuralmente invisible: mover el CoM a lo largo de la gravedad no cambia
ningún momento. Por eso el método clásico inclina la muñeca por varias orientaciones — para
que la gravedad apunte a otro sitio y esa dirección deje de ser ciega. Eso es lo que cuesta
tiempo, y es lo que este método decide no comprar.

## 3. Lo que necesitas antes de la primera lectura

**Tara de la herramienta, una vez.** Esto sí necesita orientaciones distintas, y no se puede
saltar. Con la herramienta vacía, el sensor lee
`F = -m_h·g_s + b_F` y `τ = [g_s]_× · c_h + b_τ`, con `c_h = m_h · r_h` el primer momento de
masa de la herramienta. Son dos bloques lineales: el de fuerza tiene 4 incógnitas y necesita
≥2 poses no paralelas; el de par tiene 6 y necesita ≥3 poses cuyas diferencias de gravedad
no sean colineales. Cuatro poses inclinadas van sobradas.

No cedas aquí. El error de la tara es **sistemático**: no baja promediando y se propaga casi
lineal al CoM de cada bulto. Es una calibración de arranque o de turno, no parte del ciclo,
así que que cueste diez segundos da igual.

> Atajo tentador que conviene entender antes de usarlo: si vas a medir **siempre** en la
> misma orientación, te basta una lectura con la herramienta vacía en esa orientación y
> restarla entera, sin descomponer nada. Es exacto mientras la orientación no cambie — pero
> la herramienta cargada se comba distinto que vacía, y ese par de grados de diferencia ya
> mete error. Con el modelo completo lo predices en la orientación real de cada lectura.

**La transformada del agarre.** Al sellar, guarda la posición y la rotación del objeto
respecto a la herramienta. La posición es tu *prior*; la rotación es lo que te devuelve el
resultado en los ejes del bulto.

## 4. El algoritmo

Todo el núcleo son diez líneas. `force`, `torque` y `gravity` en el frame del sensor, con la
tara ya restada:

```python
import numpy as np

def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

def com_from_one_reading(force, torque, gravity, prior, rank_tolerance=1e-3):
    """Masa y centro de masas en el frame del sensor, más la dirección que no se ha visto.

    `prior` es dónde se asume el CoM en las direcciones que la pose no resuelve: el centro
    geométrico del bulto, en el frame del sensor.
    """
    mass = np.linalg.norm(force) / np.linalg.norm(gravity)
    matrix = -skew(force)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    resolved = singular > singular[0] * rank_tolerance
    correction = right[resolved].T @ ((left[:, resolved].T @ (torque - matrix @ prior)) / singular[resolved])
    return mass, prior + correction, right[-1]
```

Dos cosas que parecen detalles y no lo son:

- **Truncar, no amortiguar.** La SVD truncada deja la parte observable *exactamente* igual
  que un ajuste de rango completo: el prior no es un peso que compite con los datos, es qué
  hacer donde no hay datos. Una regularización de Tikhonov (`ridge`) sí contamina el plano
  horizontal, que es justo lo que quieres intacto.
- **Anclar en el centro geométrico, no en cero.** La solución de norma mínima de
  `lstsq` —lo que sale si pasas `prior = 0`— pone el CoM en el origen del sensor a lo largo
  del eje ciego, o sea decenas de centímetros por encima del bulto. Con el centro geométrico
  como ancla, el error de esa componente es como mucho la semialtura de la caja.

El mismo código sirve para el método lento: si le pasas varias poses inclinadas, apiladas en
`matrix` y `torque`, las tres direcciones quedan resueltas y el prior no deja rastro. Un solo
camino de código para las dos precisiones.

Para pasarlo a los ejes del bulto, con `R` la rotación bulto→sensor y `p` el prior:

```python
com_local = R.T @ (com_sensor - p)      # medido desde el centro geométrico del bulto
```

## 5. El chivato: saber cuándo la suposición miente

Esta es la parte que no se puede omitir. Fijar una dirección solo es seguro mientras esa
dirección sea la que a nadie le importa. Si la herramienta está inclinada, el eje ciego se
sale de la vertical del bulto y el error del prior **se proyecta sobre el plano**, que sí se
lee.

El propio SVD te da el eje ciego gratis (`right[-1]`). Pásalo a los ejes del bulto y mide
cuánto se inclina:

```python
blind = R.T @ blind_axis
lean = np.hypot(blind[0], blind[1])            # seno del ángulo con la vertical del bulto
hidden = float(np.abs(blind) @ half_extents) * lean
```

`hidden` es el error en planta que esa dirección podría estar escondiendo, en el peor caso:
el desplazamiento real a lo largo del eje ciego está acotado por las semidimensiones —un
centro de masas no puede estar fuera de su propia caja— y lo que llega al plano es esa
proyección por el seno de la inclinación.

Ponle un umbral y compáralo con algo real de tu sistema. Nosotros usamos 10 mm porque el
planificador trabaja con márgenes al vuelco de 8 mm en adelante; con la herramienta a plomo
sale 1,8 mm. Por encima del umbral, la lectura se marca no fiable y el sistema puede caer al
barrido inclinado para ese bulto concreto.

## 6. Trampas que nos costaron tiempo

Por orden de cuánto duelen.

1. **Modelar la ventosa como una restricción soldada falsea el par.** En simulación, un
   `weld` mete entre 50 y 750 mm de error de CoM según la rigidez, y no es un artefacto de
   asentamiento: sobrevive a esperar lo que quieras. Un agarre sellado es una unión rígida,
   así que reconstruye la escena con el bulto emparentado a la herramienta y quita la
   restricción.
2. **La pose de lectura tiene que estar realmente a plomo.** Nuestro levantamiento de
   seguridad movía un eje de hombro 0,35 rad, lo que inclina la herramienta: el eje ciego se
   iba 20° de la vertical del cartón y metía ~45 mm en el plano. El síntoma era exactamente
   `semialtura · sin(inclinación)`. Ahora ese levantamiento solo se hace si el barrido va a
   inclinar la muñeca, que es para lo que existía.
3. **"¿Va despacio?" no es lo mismo que "¿está quieto?".** Un brazo pasando por el punto
   bajo de una oscilación cumple cualquier umbral de velocidad instantánea. Exige N pasos
   **consecutivos** por debajo del umbral. Nos costó una lectura de 1371 N donde el peso real
   no llegaba a 80.
4. **El bulto tiene que estar libre de verdad.** Si toca a un vecino, al suelo o a la caja
   que acabas de soltar, estás pesando las dos. Vimos 37 kg donde había 5,4. Lee después del
   levantamiento que ya hace el ciclo, no antes.
5. **El guardarraíl de "misma pose" al promediar hay que aflojarlo, no quitarlo.** Al
   promediar varias muestras conviene verificar que vienen de la misma orientación; con una
   lectura rápida el brazo todavía se escurre unas milésimas de radián y el umbral estricto
   salta siempre. Aflójalo a lo que sea inofensivo (nosotros ~0,3°) pero déjalo puesto: sigue
   cazando el error de mezclar dos poses distintas.
6. **La ventana de promediado es tu relación precisión/tiempo.** El ruido blanco baja con
   `√N`. Bajamos de 120 a 30 muestras (0,24 → 0,06 s) y la masa pasó de ~0 g a 21 g de error
   sobre 7,8 kg. El plano horizontal apenas se movió, porque ahí el término dominante no es
   el ruido sino la tara.

## 7. Cómo validarlo en tu celda

No te creas el método, mídelo. Lo que hace falta es correr el **mismo** trabajo dos veces,
una con cada pesaje, y comparar tres cosas:

- **El error, separado en plano y vertical.** Mezclados no dicen nada: el vertical va a
  dominar por construcción y es el que no se usa. Nosotros informamos `error_mm` y
  `error_xy_mm` por bulto.
- **El coste, en tiempo de celda por bulto**, no de reloj de CPU.
- **El resultado aguas abajo.** Esta es la prueba de verdad: ¿sale la misma pila? Si las dos
  precisiones producen las mismas colocaciones, pagar por la componente extra no compra
  nada. Nosotros comparamos la lista de colocaciones y cuántos bultos se replanificaron
  después de medir.

Aquí, sobre ocho cartones (`stable-pallet run benchmark-probe`):

| | barrido de 4 poses | una lectura en el sitio |
|---|---|---|
| Coste por bulto | 13,400 s | **0,108 s** (124× menos) |
| Error de CoM en planta | 0,00 mm | 0,13 mm medio, 0,29 máx. |
| Error de CoM en 3D | 0,00 mm | 12,1 mm medio, 36,1 máx. (todo vertical) |
| Error de masa | ~0 g | 21 g sobre 7,8 kg |
| Bultos replanificados tras medir | 5 de 8 | 5 de 8 |
| Pila resultante | estable, apoyo 73 %, margen 80 mm | idéntica |

## 8. Qué cambia en hardware real

- La tara pasa a ser de turno, y conviene repetirla si cambia la temperatura: la deriva del
  transductor entra justo por donde entra la tara.
- El ruido del sensor deja de ser numérico. Mide su `σ` con la herramienta quieta y elige la
  ventana de promediado con `σ/√N` contra el error de CoM que te puedes permitir, que para un
  bulto de masa `m` es `σ_τ / (m·g)`. Ojo: **el error de CoM escala con 1/m**, así que el
  bulto ligero es el caso difícil, no el pesado.
- Si el agarre puede resbalar, has perdido el detector que comparaba el módulo de la fuerza
  entre poses. Recupéralo con dos lecturas dentro de un movimiento que ya hagas —no dan
  información nueva de `r`, pero dos módulos de fuerza distintos delatan que algo se movió.
- Si la pose de agarre viene de una cámara, el prior hereda su error. Sigue siendo mejor
  ancla que cero por un orden de magnitud.

## 9. Cuándo no usarlo

- Si algo aguas abajo lee la componente vertical del CoM: apilado con vuelco dinámico,
  transporte con frenadas fuertes, o control de un brazo que compense inercias.
- Si el objeto no es rígido. Un saco redistribuye la masa al moverlo, y entonces ni la
  lectura única ni el barrido describen lo mismo dos veces.
- Si no puedes garantizar la orientación de la herramienta en el momento de leer. Sin eso,
  el chivato de la sección 5 se dispara constantemente y no ganas nada.

## 10. La idea general, más allá del CoM

El patrón es reutilizable y merece la pena tenerlo a mano: **antes de pagar por observar una
magnitud, mira si quien la consume la usa.** Aquí el sistema era de rango deficiente, la
solución tradicional era gastar movimiento para subir el rango, y resultó que la dirección
que faltaba era exactamente la que nadie leía. Buscar esa coincidencia —la dirección ciega
del sensor alineada con la dirección irrelevante del consumidor— es lo que convierte trece
segundos en una décima.

---

Implementación en este repositorio: `tools/stable_pallet/com_estimator.py` (el despeje),
`tools/stable_pallet/com_probe.py` (las poses, el prior y el chivato),
`tools/stable_pallet/tare.py` (la calibración de la herramienta) y
`tools/stable_pallet/wrench.py` (el modelo del sensor). En la celda,
`src/vision/gauge.py` usa el mismo despeje. El barrido completo sigue disponible con
`--precise-com`.
