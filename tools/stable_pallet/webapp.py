"""Browser control panel for the palletizing demonstrator.

Pick a level, press play and watch it in the MuJoCo window. Every switch is read once,
when the run starts, and travels as a flag on the command line: `scripts/palletize.py`
reports over a one-way pipe, so nothing can be changed once it is running. The transport
bar and the live switches of the old local runner are disabled in the page for that
reason -- see the `disabled` marks in `web/index.html`.

Three processes, each with one job. Both modes launch the migrated entrypoint with the
selected source and level. Execution may upload; debugging always passes
`--no-telemetry` and never opens a remote episode:

    browser  <--HTTP/SSE-->  this server  <--JSON over pipes-->  scripts/palletize.py

Nothing in this module touches MuJoCo. Of the control messages the page can send only
`cancel` means anything -- it stops the child -- and everything the child reports is
republished to every page that is listening, which is why the handlers below are all a
few lines long: the worst any of them can do is put a dict on a queue.

The server is stdlib only -- no framework, no bundler, no network at start-up -- so the
demonstrator keeps working on a laptop with the wifi switched off.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import signal
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from .controls import SPEED_PRESETS
from .runner import interpreter, is_viewer_noise

STATIC = Path(__file__).parent / "web"
REPO = Path(__file__).resolve().parents[2]
# A listener that has not been drained in this many events is a closed tab the socket
# has not noticed yet. Dropping it is better than growing its queue forever.
LISTENER_BACKLOG = 512

# Long enough to stay out of the way, short enough that a proxy or a suspended laptop
# does not silently kill the stream.
HEARTBEAT_SECONDS = 15.0


def _idle_state() -> dict[str, Any]:
    return {
        "frames": 0,
        "index": None,
        "paused": False,
        "playback": "stopped",
        "holding": False,
        "activity": "",
    }


def _catalogue() -> list[dict[str, Any]]:
    """Los nueve niveles declarados en YAML, agrupables por fuente."""
    config = yaml.safe_load((REPO / "configs" / "pallet.yaml").read_text(encoding="utf-8"))
    descriptions = {
        "table": "Bultos preparados en mesa.",
        "conveyor": "Banda física que entrega y se detiene.",
        "truck": "Descarga del remolque de arriba abajo.",
    }
    return [
        {
            "key": f"level-{row['id']}",
            "title": row["name"],
            "description": descriptions[row["source"]],
            "kind": "level",
            "source": row["source"],
            "level": int(row["id"]),
            "watchable": True,
            "usesRobot": True,
            "shake": False,
            "instantPlace": False,
            "seed": 1,
        }
        for row in config["levels"]
    ]


def _palletize_argv(request: dict[str, Any]) -> list[str]:
    """La línea de `scripts/palletize.py` para el nivel elegido.

    DEPURACIÓN añade `--no-telemetry` para que el entrypoint no abra subida. La puerta
    HTTP (`_run_request`) siempre manda `mode`, y un `POST /api/run` sin él sube: es la
    regla «se sube por defecto» de `AGENTS.md` §5. El `debug` de aquí sólo cubre a quien
    construya la petición a mano.
    """
    python = REPO / ".venv" / "bin" / "python"
    argv = [
        interpreter(bool(request.get("viewer")), python if python.exists() else None),
        str(REPO / "scripts" / "palletize.py"),
        "--protocol", "json",
        "--source", str(request["source"]),
        "--level", str(request["level"]),
        "-n", "1",
    ]
    if str(request.get("mode", "debug")) != "execution":
        argv.append("--no-telemetry")
    if request.get("viewer"):
        argv.append("--viewer")
    if request.get("simplified_graphics"):
        argv.append("--simplified-graphics")
    if request.get("show_com"):
        argv.append("--show-com")
    if request.get("stability_test"):
        argv.append("--stability-test")
    argv.extend(("--speed", "0" if request.get("fast_forward") else str(request["speed"])))
    if request.get("seed") is not None:
        argv.extend(("--seed", str(int(request["seed"]))))
    return argv


def unwind_child(process: subprocess.Popen[Any], timeout: float = 5.0) -> None:
    """Pide al hijo que deshaga. ``terminate()`` (SIGTERM) no recorre ``finally``.

    En POSIX, SIGINT se convierte en ``KeyboardInterrupt`` y sí cierra el episodio.
    SIGTERM y kill quedan como respaldo si el proceso no sale.
    """
    if process.poll() is not None:
        return
    sigint = getattr(signal, "SIGINT", None)
    if sigint is not None:
        try:
            process.send_signal(sigint)
        except (ProcessLookupError, OSError, ValueError):
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()


class PalletizeClient:
    """El entrypoint real, visto con el mismo protocolo de líneas que el runner local."""

    def __init__(
        self,
        request: dict[str, Any],
        on_message: Any,
        *,
        argv: list[str] | None = None,
    ) -> None:
        # `argv` es la costura de los tests: inyecta un hijo falso sin tocar la petición.
        if argv is None:
            argv = _palletize_argv(request)
        self.on_message = on_message
        self.process = subprocess.Popen(
            argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read(self, pipe) -> None:
        for line in iter(pipe.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                self.on_message(json.loads(line))
            except ValueError:
                self.on_message({"kind": "log", "text": line})
        self.on_message({"kind": "closed"})

    def _read_errors(self) -> None:
        for line in iter(self.process.stderr.readline, ""):
            line = line.rstrip()
            if line and not is_viewer_noise(line):
                self.on_message({"kind": "log", "text": line})

    def is_running(self) -> bool:
        return self.process.poll() is None

    def send(self, **message: Any) -> None:
        if message.get("command") == "cancel":
            self.stop()

    def stop(self, timeout: float = 5.0) -> None:
        """SIGINT para que el CLI cierre el episodio; SIGTERM y kill si no sale."""
        unwind_child(self.process, timeout)


class Session:
    """The one run the page is driving, and everybody watching it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: set[queue.Queue[dict[str, Any]]] = set()
        self.client: PalletizeClient | None = None
        self.state = _idle_state()
        self.log: list[dict[str, Any]] = []
        self.title: str = ""
        self.mode: str = "debug"

    # -- listeners -------------------------------------------------------------------

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        """Register a page and hand it everything it missed."""
        stream: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._listeners.add(stream)
            stream.put(
                {
                    "kind": "hello",
                    "running": self.running,
                    "state": dict(self.state),
                    "log": list(self.log),
                    "title": self.title,
                    "mode": self.mode,
                }
            )
        return stream

    def unsubscribe(self, stream: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._listeners.discard(stream)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            stale = [stream for stream in self._listeners if stream.qsize() > LISTENER_BACKLOG]
            for stream in stale:
                self._listeners.discard(stream)
            for stream in self._listeners:
                stream.put(event)

    # -- the run ---------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.client is not None and self.client.is_running()

    def start(self, request: dict[str, Any], title: str) -> None:
        if self.running:
            raise RuntimeError("Ya hay un experimento en marcha.")
        self.state = _idle_state()
        # Kept rather than only announced, so a page opened mid-run still knows what it
        # is looking at.
        self.log = [{"text": f"\u25b6 {title}", "tone": "head"}]
        self.title = title
        self.mode = str(request.get("mode", "debug"))
        self.publish({"kind": "started", "title": title, "mode": self.mode})
        self.client = PalletizeClient(request, self._on_runner_message)

    def send(self, message: dict[str, Any]) -> None:
        if self.client is not None and self.client.is_running():
            self.client.send(**message)

    def stop(self) -> None:
        if self.client is not None and self.client.is_running():
            self.client.stop()

    def _on_runner_message(self, message: dict[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "state":
            self.state = {key: message[key] for key in _idle_state() if key in message}
        elif kind == "log":
            self._remember({"text": message.get("text", ""), "tone": "note"})
        elif kind == "finished":
            tone = "ok" if message.get("ok") else "bad"
            for index, line in enumerate(message.get("lines", [])):
                self._remember({"text": line, "tone": tone if index == 0 else "note"})
        elif kind == "closed":
            self.state = _idle_state()
        self.publish(message)

    def _remember(self, line: dict[str, Any]) -> None:
        self.log.append(line)
        del self.log[:-400]


def _run_request(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Traduce la tarjeta y el modo que eligió la página a la petición del entrypoint.

    Los dos modos llevan `source` y `level`. DEPURACIÓN no elige un experimento legado:
    `PalletizeClient` añade `--no-telemetry` según `mode`. Cualquiera de las dos casillas
    de centro de masa enciende `--show-com`: el entrypoint sólo tiene esa bandera, y
    pinta el CoG final del palé en el informe. El ensayo de estabilidad va en los dos
    modos y se anuncia en el título, que es lo único que distingue un run con ensayo.
    """
    item = next((entry for entry in _catalogue() if entry["key"] == payload.get("experiment")), None)
    if item is None:
        raise ValueError(f"Experimento desconocido: {payload.get('experiment')!r}")
    mode = str(payload.get("mode", "execution"))
    if mode not in {"execution", "debug"}:
        raise ValueError(f"Modo desconocido: {mode!r}")
    viewer = bool(payload.get("viewer")) and item["watchable"]
    request = {
        "mode": mode,
        "source": item["source"],
        "level": item["level"],
        "viewer": viewer,
        "speed": float(payload.get("speed", 1.0)),
        "fast_forward": bool(payload.get("fast_forward")),
        "simplified_graphics": bool(payload.get("simplified_graphics")),
        "show_com": bool(payload.get("show_true_com")) or bool(payload.get("show_estimated_com")),
        "stability_test": bool(payload.get("stability_test")),
        "seed": payload.get("seed"),
    }
    title = f"{'EJECUCIÓN' if mode == 'execution' else 'DEPURACIÓN'} · {item['title']}"
    if request["stability_test"]:
        title += " · ESTABILIDAD"
    return request, title


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "StablePallet"

    @property
    def session(self) -> Session:
        return self.server.session  # type: ignore[attr-defined]

    def log_message(self, *_args: Any) -> None:
        """Keep the console for the experiment log, not for request lines."""

    # -- routing ----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - the name is the stdlib's
        route = self.path.split("?", 1)[0]
        if route == "/api/experiments":
            self._json({"experiments": _catalogue(), "speeds": [list(pair) for pair in SPEED_PRESETS],
                        "modes": ["execution", "debug"]})
        elif route == "/api/events":
            self._events()
        else:
            self._static(route)

    def do_POST(self) -> None:  # noqa: N802 - the name is the stdlib's
        route = self.path.split("?", 1)[0]
        try:
            payload = self._body()
            if route == "/api/run":
                self._run(payload)
            elif route == "/api/control":
                self.session.send(payload)
                self._json({"ok": True})
            elif route == "/api/stop":
                self.session.stop()
                self._json({"ok": True})
            else:
                self._json({"error": "Ruta desconocida"}, status=404)
        except (RuntimeError, ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, status=400)

    # -- handlers ----------------------------------------------------------------------

    def _run(self, payload: dict[str, Any]) -> None:
        request, title = _run_request(payload)
        self.session.start(request, title)
        self._json({"ok": True})

    def _events(self) -> None:
        stream = self.session.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        # No length is possible on a stream, so the connection itself delimits the body.
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                try:
                    event = stream.get(timeout=HEARTBEAT_SECONDS)
                    chunk = f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    chunk = ": keep-alive\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the tab went away
        finally:
            self.session.unsubscribe(stream)

    def _static(self, route: str) -> None:
        name = "index.html" if route == "/" else route.lstrip("/")
        path = (STATIC / name).resolve()
        if not path.is_file() or STATIC.resolve() not in path.parents:
            self._json({"error": "No encontrado"}, status=404)
            return
        body = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._respond(body, kind)

    # -- plumbing ------------------------------------------------------------------------

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._respond(json.dumps(payload).encode(), "application/json", status)

    def _respond(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], session: Session) -> None:
        super().__init__(address, _Handler)
        self.session = session


def launch(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True) -> int:
    """Serve the panel until interrupted."""
    session = Session()
    try:
        server = _Server((host, port), session)
    except OSError as exc:
        print(f"No se pudo abrir {host}:{port}: {exc}")
        print("Prueba con otro puerto: `stable-pallet dashboard --port 8123`.")
        return 1

    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Panel en {url}")
    print("Ctrl-C para cerrarlo.")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando…")
    finally:
        session.stop()
        server.shutdown()
        server.server_close()
    return 0
