"""
Las vistas del palé que se suben a la plataforma.

PORTAR `render(scene, view)` desde `~/HackSpain/paletizado-guionizado/src/pallet/scene.py`
(está al final del fichero). El renderer se crea y se destruye en cada llamada: esto
pasa unas pocas veces por episodio y no compensa mantener vivo un contexto GL entre
medias. En percepción es al revés, y por eso vive en otro sitio.

**`view` es un vocabulario cerrado: `top | side | iso | camera`.** Con otro nombre el
PNG sube a Storage y **luego** la base rechaza la fila con un `23514`: la foto queda
huérfana y la traza sin imagen. Al repo anterior le pasó con `front`. Las cámaras de
`configs/pallet.yaml` tienen que llamarse con uno de esos cuatro nombres, y hay un test
que lo comprueba sin arrancar el simulador.

Y lo de siempre con las fotos: **el brazo se aparta primero, y en cartesiano**. Acaba
justo encima del palé, que es donde estaba soltando, así que sin apartarlo la cenital
sale del dorso de la mano. Y apartarlo en espacio de juntas barre el montón recién
colocado: medido, el episodio pasaba de 10/10 a 4/10 con `overhang_violation` en cuanto
se metió la foto por capa. La maniobra vive en `episode.py`; aquí solo se dispara.
"""
