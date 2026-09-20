"""Run one experiment in its own process, driven over a line protocol.

The MuJoCo viewer owns a window, a GL context and a render thread of its own, and it
expects to be the only thing in the process doing that. Giving it one keeps the panel
free to serve pages, and makes stopping a run a matter of closing a pipe rather than
unpicking threads.

The two exchange newline-delimited JSON: the panel sends control changes and transport
commands, the runner reports how far the recording has got and what it found.

`ViewerControls` still does the actual work on the runner side -- the protocol only
moves the same field assignments across a pipe.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .controls import RunCancelled, ViewerControls
from .experiments import RunOptions, experiment, run_experiment, summarise

REPORT_INTERVAL = 0.1
LIVE_FIELDS = ("speed", "fast_forward", "show_true_com", "show_estimated_com")

# GLFW warns once per call about everything the compositor does not implement -- window
# position, decorations -- and under Wayland that is several lines a second. None of it
# is actionable, and it buries the run's own output. Anything else on stderr, including
# tracebacks, still reaches the log.
VIEWER_NOISE = ("GLFWError", "warnings.warn", "libdecor", "Wayland: The platform")


# -- runner side -------------------------------------------------------------------


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def apply_message(controls: ViewerControls, message: dict[str, Any]) -> None:
    """Turn one protocol message into control changes."""
    for field in LIVE_FIELDS:
        if field in message:
            setattr(controls, field, message[field])
    command = message.get("command")
    if command == "pause":
        controls.paused = True
    elif command == "resume":
        controls.resume()
    elif command == "cancel":
        controls.cancel()
    elif command == "release":
        controls.holding = False
    elif command == "scrub":
        controls.scrub_to(_clamp(controls, int(message["index"])))
    elif command == "play":
        controls.play(str(message["direction"]))
    elif command == "step":
        current = controls.review_index
        if current is None:
            current = controls.frame_count - 1
        controls.scrub_to(_clamp(controls, current + int(message["delta"])))


def _clamp(controls: ViewerControls, index: int) -> int:
    return max(0, min(index, max(0, controls.frame_count - 1)))


def is_viewer_noise(line: str) -> bool:
    """Whether a stderr line is the viewer complaining about the compositor."""
    return any(token in line for token in VIEWER_NOISE)


def _report(controls: ViewerControls) -> dict[str, Any]:
    return {
        "kind": "state",
        "frames": controls.frame_count,
        "index": controls.review_index,
        "paused": controls.paused,
        "playback": controls.playback,
        "holding": controls.holding,
        "activity": controls.activity,
    }


def _listen(controls: ViewerControls) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            apply_message(controls, json.loads(line))
        except (ValueError, KeyError) as exc:
            emit({"kind": "log", "text": f"Mensaje ignorado: {exc}"})


def _report_loop(controls: ViewerControls, done: threading.Event) -> None:
    while not done.wait(REPORT_INTERVAL):
        emit(_report(controls))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    request = json.loads(argv[0])
    item = experiment(request["experiment"])
    controls = ViewerControls(**request.get("controls", {}))
    controls.start_run(hold_at_end=bool(request.get("hold_at_end")))

    threading.Thread(target=_listen, args=(controls,), daemon=True).start()
    done = threading.Event()
    threading.Thread(target=_report_loop, args=(controls, done), daemon=True).start()

    options = RunOptions(
        viewer=bool(request.get("viewer")) and item.watchable,
        controls=controls,
        simplified_graphics=bool(request.get("simplified_graphics")),
        measure_com=request.get("measure_com"),
        seed=request.get("seed"),
    )
    started = time.monotonic()
    try:
        result = run_experiment(item, options)
    except RunCancelled:
        lines, ok = ["Detenido por el operador."], False
    except Exception as exc:  # noqa: BLE001 - the panel is the only place to report this
        lines, ok = [f"{type(exc).__name__}: {exc}"], False
    else:
        ok = bool(result.get("success", True))
        lines = [
            f"Terminado en {time.monotonic() - started:.1f} s",
            *summarise(item, result),
            f"Artefacto: {result['artifact']}",
        ]
    done.set()
    emit({"kind": "finished", "ok": ok, "lines": lines})
    return 0 if ok else 2


# -- panel side --------------------------------------------------------------------


def interpreter(viewer: bool, python: Path | str | None = None) -> str:
    """The interpreter a child process should run under.

    macOS only lets MuJoCo's passive viewer run from `mjpython`, a launcher that gives the
    window the main thread. It sits next to `python` in the venv's `bin/`, so a run that
    wants the window takes that one when it is there. Anywhere else it is plain `python`."""
    python = Path(python or sys.executable)
    if viewer and sys.platform == "darwin":
        launcher = python.parent / "mjpython"
        if launcher.exists():
            return str(launcher)
    return str(python)


class RunnerClient:
    """A running experiment, seen from the panel."""

    def __init__(self, request: dict[str, Any], on_message: Callable[[dict[str, Any]], None]) -> None:
        self.on_message = on_message
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv, only the request varies
            [interpreter(bool(request.get("viewer"))), "-m", "stable_pallet.runner", json.dumps(request)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_reports, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read_reports(self) -> None:
        # `for line in pipe` reads ahead, which holds a report back until enough of them
        # pile up to fill a buffer. The panel would show a frozen transport bar.
        for line in iter(self.process.stdout.readline, ""):
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

    def send(self, **message: Any) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def is_running(self) -> bool:
        return self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the run to stop, and insist if it will not."""
        self.send(command="cancel")
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
