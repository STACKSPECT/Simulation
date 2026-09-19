#!/usr/bin/env bash
# venv + dependencias + SDK de la plataforma + modelo del brazo.
#
# El SDK `theker_telemetry` NO vive en este repo: es el contrato de lo que la plataforma
# guarda, y quien define la forma del dato es quien lo almacena. Se instala editable
# desde el repo Platform, y tiene que ser la rama `dev`: en `main` no está `pallet.py`
# (de donde salen `stability_margin` y `support_polygon`) y el ciclo de vida del episodio
# tampoco. El fallo no aparece al instalar, sino al correr — por eso se comprueba aquí.
set -euo pipefail

cd "$(dirname "$0")/.."
PLATFORM="${PLATFORM:-../Platform}"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ -d "$PLATFORM/backend" ]; then
    .venv/bin/pip install -e "$PLATFORM/backend"
else
    echo "aviso: no encuentro $PLATFORM/backend — pasa PLATFORM=<ruta al repo Platform>"
    exit 1
fi

.venv/bin/python - <<'PY'
from theker_telemetry import RunLog, stability_margin, support_polygon  # noqa: F401
for m in ("begin", "end", "snapshot"):
    assert hasattr(RunLog, m), f"SDK sin {m}(): necesitas la rama dev de Platform"
print("SDK ok: ciclo de vida y geometría disponibles")
PY

# El modelo del brazo sale del Menagerie. Cuál, lo dice `configs/scene.yaml`:
# `robot.model` apunta a un `scene.xml` de aquí dentro.
if [ ! -d third_party/mujoco_menagerie ]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie \
        third_party/mujoco_menagerie
fi

echo
echo "listo. source .venv/bin/activate"
echo "credenciales: copia .env.example a .env (o ponlo en la carpeta padre)"
