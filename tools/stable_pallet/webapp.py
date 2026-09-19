"""Browser control panel for the palletizing demonstrator.

Pick a prepared experiment, press play and watch it in the MuJoCo window. While it runs
the page keeps the switches that are safe to change live -- speed, fast-forward, the
centre-of-mass markers -- and a transport bar that pauses the cell and steps back
through what has already happened.

Three processes, each with one job:

    browser  <--HTTP/SSE-->  this server  <--JSON over pipes-->  stable_pallet.runner

Nothing in this module touches MuJoCo. It forwards control messages to the runner and
republishes whatever the runner reports to every page that is listening, which is why
the handlers below are all a few lines long: the worst any of them can do is put a dict
on a queue.

The server is stdlib only -- no framework, no bundler, no network at start-up -- so the
demonstrator keeps working on a laptop with the wifi switched off.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .controls import SPEED_PRESETS
from .experiments import EXPERIMENTS
from .runner import RunnerClient

STATIC = Path(__file__).parent / "web"
FRAME_SECONDS = 0.05

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
    """The experiment list, as the page needs it."""
    return [
        {
            "key": item.key,
            "title": item.title,
            "description": item.description,
            "kind": item.kind,
            "source": item.source,
            "watchable": item.watchable,
            "usesRobot": item.uses_robot,
            "shake": item.shake,
            "instantPlace": item.instant_place,
            "seed": item.seed,
        }
        for item in EXPERIMENTS
    ]


class Session:
    """The one run the page is driving, and everybody watching it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: set[queue.Queue[dict[str, Any]]] = set()
        self.client: RunnerClient | None = None
        self.state = _idle_state()
        self.log: list[dict[str, Any]] = []
        self.title: str = ""

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
        self.publish({"kind": "started", "title": title})
        self.client = RunnerClient(request, self._on_runner_message)

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
            self._json({"experiments": _catalogue(), "speeds": [list(pair) for pair in SPEED_PRESETS]})
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
        item = next((entry for entry in EXPERIMENTS if entry.key == payload.get("experiment")), None)
        if item is None:
            raise ValueError(f"Experimento desconocido: {payload.get('experiment')!r}")
        viewer = bool(payload.get("viewer")) and item.watchable
        request = {
            "experiment": item.key,
            "controls": {
                "speed": float(payload.get("speed", 1.0)),
                "fast_forward": bool(payload.get("fast_forward")),
                "show_true_com": bool(payload.get("show_true_com")),
                "show_estimated_com": bool(payload.get("show_estimated_com")),
            },
            "viewer": viewer,
            "hold_at_end": bool(payload.get("hold_at_end")) and viewer,
            "simplified_graphics": bool(payload.get("simplified_graphics")),
            "measure_com": bool(payload.get("measure_com")) if item.uses_robot else None,
            "seed": payload.get("seed"),
        }
        self.session.start(request, item.title)
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
