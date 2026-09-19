"""The panel server: what it tells a page, and what it asks the runner for."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from stable_pallet import runner as runner_module
from stable_pallet import webapp
from stable_pallet.runner import interpreter, is_viewer_noise
from stable_pallet.webapp import LISTENER_BACKLOG, Session, _catalogue, _Server


class FakeRunner:
    """A `RunnerClient` that records instead of starting a process."""

    instances: list[FakeRunner] = []

    def __init__(self, request: dict[str, Any], on_message: Any) -> None:
        self.request = request
        self.on_message = on_message
        self.sent: list[dict[str, Any]] = []
        self.alive = True
        FakeRunner.instances.append(self)

    def is_running(self) -> bool:
        return self.alive

    def send(self, **message: Any) -> None:
        self.sent.append(message)

    def stop(self, timeout: float = 5.0) -> None:
        self.alive = False


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunner]:
    FakeRunner.instances = []
    monkeypatch.setattr(webapp, "RunnerClient", FakeRunner)
    monkeypatch.setattr(webapp, "PalletizeClient", FakeRunner)
    return FakeRunner


@pytest.fixture
def served(runner: type[FakeRunner]) -> Any:
    """A real server on a free port, with the runner stubbed out."""
    session = Session()
    server = _Server(("127.0.0.1", 0), session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, session
    finally:
        server.shutdown()
        server.server_close()


def get(base: str, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return response.status, response.read()


def post(base: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# -- the catalogue ---------------------------------------------------------------------


def test_the_catalogue_offers_the_nine_declared_levels() -> None:
    catalogue = _catalogue()
    assert [item["key"] for item in catalogue] == [
        "level-11", "level-12", "level-13", "level-21", "level-22",
        "level-23", "level-31", "level-32", "level-33",
    ]
    for entry in catalogue:
        assert entry["title"] and entry["description"]
        assert isinstance(entry["watchable"], bool)
        assert isinstance(entry["usesRobot"], bool)


def test_the_catalogue_says_where_the_cartons_come_from() -> None:
    """The card is tagged from this, which is the only sign a run starts at a trailer."""
    sources = {entry["key"]: entry["source"] for entry in _catalogue()}
    assert sources["level-11"] == "table"
    assert sources["level-21"] == "conveyor"
    assert sources["level-31"] == "truck"
    assert set(sources.values()) == {"table", "conveyor", "truck"}


def test_every_level_has_a_watchable_cell() -> None:
    assert all(entry["watchable"] and entry["usesRobot"] for entry in _catalogue())


# -- what a page is told ------------------------------------------------------------------


def test_a_page_that_opens_late_is_told_what_is_already_running(runner: type[FakeRunner]) -> None:
    session = Session()
    session.start({"experiment": "plan"}, "Solo planificación")
    session._on_runner_message(
        {"kind": "state", "frames": 12, "index": None, "paused": True, "playback": "stopped", "holding": False}
    )

    hello = session.subscribe().get_nowait()

    assert hello["kind"] == "hello"
    assert hello["running"] is True
    assert hello["title"] == "Solo planificación"
    assert hello["state"]["frames"] == 12
    assert hello["state"]["paused"] is True
    assert hello["log"][0]["text"].endswith("Solo planificación")


def test_every_open_page_gets_every_runner_message(runner: type[FakeRunner]) -> None:
    session = Session()
    first, second = session.subscribe(), session.subscribe()
    first.get_nowait(), second.get_nowait()  # the hello each was greeted with

    session._on_runner_message({"kind": "log", "text": "colocando 3/8"})

    assert first.get_nowait()["text"] == "colocando 3/8"
    assert second.get_nowait()["text"] == "colocando 3/8"


def test_what_a_run_reported_survives_for_a_page_opened_afterwards(runner: type[FakeRunner]) -> None:
    """The summary is the point of the run; reloading the tab must not lose it."""
    session = Session()
    session.start({"experiment": "plan"}, "Solo planificación")
    session._on_runner_message({"kind": "finished", "ok": True, "lines": ["Terminado en 9.6 s", "Colocadas 8/8"]})

    log = session.subscribe().get_nowait()["log"]

    assert [line["text"] for line in log[1:]] == ["Terminado en 9.6 s", "Colocadas 8/8"]
    assert log[1]["tone"] == "ok"


def test_a_failed_run_is_marked_as_such(runner: type[FakeRunner]) -> None:
    session = Session()
    session.start({"experiment": "plan"}, "Solo planificación")
    session._on_runner_message({"kind": "finished", "ok": False, "lines": ["Detenido por el operador."]})

    assert session.log[1]["tone"] == "bad"


def test_a_page_that_stopped_reading_is_dropped(runner: type[FakeRunner]) -> None:
    """A closed tab the socket has not noticed must not grow a queue forever."""
    session = Session()
    abandoned = session.subscribe()

    for index in range(LISTENER_BACKLOG + 5):
        session.publish({"kind": "log", "text": str(index)})

    assert abandoned.qsize() <= LISTENER_BACKLOG + 2
    session.publish({"kind": "log", "text": "after"})
    assert session._listeners == set()


# -- driving a run ---------------------------------------------------------------------------


def test_a_second_run_is_refused_while_one_is_going(runner: type[FakeRunner]) -> None:
    session = Session()
    session.start({"experiment": "plan"}, "Solo planificación")

    with pytest.raises(RuntimeError):
        session.start({"experiment": "palletize"}, "Paletizado")


def test_the_page_and_its_assets_are_served(served: Any) -> None:
    base, _ = served
    status, body = get(base, "/")
    assert status == 200
    assert b"STACKSPECT" in body
    for asset in ("/app.css", "/app.js", "/tokens.css", "/brand/mark.png", "/brand/wordmark.png",
                  "/favicon.ico", "/icon.png", "/fonts/PlusJakartaSans-latin.woff2",
                  "/fonts/GeistMono-latin.woff2"):
        assert get(base, asset)[0] == 200, asset


def test_the_catalogue_is_served_with_the_speed_presets(served: Any) -> None:
    base, _ = served
    _, body = get(base, "/api/experiments")
    payload = json.loads(body)
    assert len(payload["experiments"]) == 9
    assert ["x1", 1.0] in payload["speeds"]
    assert payload["modes"] == ["execution", "debug"]


def test_a_run_reaches_the_runner_with_the_settings_the_page_chose(served: Any, runner: type[FakeRunner]) -> None:
    base, _ = served
    status, body = post(
        base,
        "/api/run",
        {
            "experiment": "level-11",
            "mode": "execution",
            "speed": 2.0,
            "fast_forward": True,
            "show_true_com": True,
            "show_estimated_com": False,
            "viewer": True,
            "hold_at_end": True,
            "simplified_graphics": True,
            "measure_com": False,
            "seed": 11,
        },
    )

    assert (status, body) == (200, {"ok": True})
    request = runner.instances[-1].request
    assert request["mode"] == "execution"
    assert request["source"] == "table"
    assert request["level"] == 11
    assert request["speed"] == 2.0
    assert request["viewer"] is True
    assert request["fast_forward"] is True
    assert request["seed"] == 11


def test_debug_mode_uses_the_local_runner_without_telemetry(served: Any, runner: type[FakeRunner]) -> None:
    base, _ = served
    post(base, "/api/run", {
        "experiment": "level-11", "mode": "debug", "viewer": True,
        "hold_at_end": True, "measure_com": False,
    })

    request = runner.instances[-1].request
    assert request["mode"] == "debug"
    assert request["experiment"] == "palletize"
    assert request["viewer"] is True
    assert request["hold_at_end"] is True
    assert request["measure_com"] is False


def test_an_unknown_experiment_is_refused(served: Any) -> None:
    base, _ = served
    status, body = post(base, "/api/run", {"experiment": "no-such-thing"})
    assert status == 400
    assert "no-such-thing" in body["error"]


def test_transport_commands_are_passed_straight_through(served: Any, runner: type[FakeRunner]) -> None:
    base, _ = served
    post(base, "/api/run", {"experiment": "level-11", "mode": "debug"})
    post(base, "/api/control", {"command": "scrub", "index": 7})
    post(base, "/api/control", {"speed": 0.5})

    assert runner.instances[-1].sent == [{"command": "scrub", "index": 7}, {"speed": 0.5}]


def test_stopping_asks_the_runner_to_stop(served: Any, runner: type[FakeRunner]) -> None:
    base, _ = served
    post(base, "/api/run", {"experiment": "level-11", "mode": "debug"})
    post(base, "/api/stop", {})

    assert runner.instances[-1].alive is False


def test_what_the_cell_is_busy_with_reaches_the_page(runner: type[FakeRunner]) -> None:
    """Nothing steps while the planner thinks, so the pill is all the page has to show."""
    session = Session()
    session.start({"experiment": "palletize"}, "Paletizado")
    stream = session.subscribe()
    stream.get_nowait()

    session._on_runner_message({"kind": "state", "frames": 40, "activity": "Planificando la caja 5/8…"})

    assert stream.get_nowait()["activity"] == "Planificando la caja 5/8…"
    assert session.state["activity"] == "Planificando la caja 5/8…"
    assert session.subscribe().get_nowait()["state"]["activity"] == "Planificando la caja 5/8…"


def test_a_finished_run_is_not_still_claiming_to_be_busy(runner: type[FakeRunner]) -> None:
    session = Session()
    session.start({"experiment": "palletize"}, "Paletizado")
    session._on_runner_message({"kind": "state", "frames": 40, "activity": "Planificando…"})
    session._on_runner_message({"kind": "closed"})

    assert session.state["activity"] == ""


# -- the log ---------------------------------------------------------------------------------


def test_the_compositor_complaints_are_kept_out_of_the_log() -> None:
    """GLFW warns several times a second under Wayland and buries the run's output."""
    assert is_viewer_noise("/x/glfw/__init__.py:917: GLFWError: (65548) b'Wayland: ...'")
    assert is_viewer_noise("  warnings.warn(message, GLFWError)")
    assert is_viewer_noise("Failed to load plugin 'libdecor-gtk.so': failed to init")


def test_a_real_error_still_reaches_the_log() -> None:
    assert not is_viewer_noise("Traceback (most recent call last):")
    assert not is_viewer_noise("ValueError: no stable placement for package 4")


@pytest.fixture
def venv_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "mjpython"):
        (bin_dir / name).touch()
    return bin_dir


def test_on_macos_a_run_with_a_window_goes_through_mjpython(venv_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert interpreter(True, venv_bin / "python") == str(venv_bin / "mjpython")


def test_a_run_without_a_window_or_off_macos_keeps_plain_python(venv_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert interpreter(False, venv_bin / "python") == str(venv_bin / "python")
    monkeypatch.setattr("sys.platform", "linux")
    assert interpreter(True, venv_bin / "python") == str(venv_bin / "python")


def test_macos_without_mjpython_falls_back_to_python(venv_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (venv_bin / "mjpython").unlink()
    monkeypatch.setattr("sys.platform", "darwin")
    assert interpreter(True, venv_bin / "python") == str(venv_bin / "python")


def test_both_launchers_ask_for_the_window_interpreter(venv_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        stdout = stderr = iter(())

        def poll(self) -> int:
            return 0

    def fake_popen(argv: list[str], **_: Any) -> FakeProcess:
        started.append(argv)
        return FakeProcess()

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("sys.executable", str(venv_bin / "python"))
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(runner_module.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(webapp, "REPO", venv_bin.parent)

    runner_module.RunnerClient({"viewer": True}, lambda message: None)
    webapp.PalletizeClient({"source": "table", "level": 11, "speed": 1.0, "viewer": True}, lambda message: None)
    runner_module.RunnerClient({"viewer": False}, lambda message: None)

    assert [argv[0] for argv in started] == [str(venv_bin / "mjpython")] * 2 + [str(venv_bin / "python")]
