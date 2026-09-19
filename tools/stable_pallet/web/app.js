/* The panel, client side.
 *
 * Settings live in the widgets; the only state worth holding here is what the runner
 * last said about itself. Every control either posts one message or re-renders from
 * that state, so there is no place for the page and the cell to disagree for long. */

const FRAME_SECONDS = 0.05;

const SPEED_LABELS = { "x0.5": "\u00d70,5", x1: "\u00d71", x2: "\u00d72", x4: "\u00d74", max: "m\u00e1x." };

const $ = (id) => document.getElementById(id);

const ui = {
  main: document.querySelector("main"),
  status: $("status"),
  link: $("link"),
  modeBadge: $("mode-badge"),
  modes: $("modes"),
  modeNote: $("mode-note"),
  experiments: $("experiments"),
  experimentCount: $("experiment-count"),
  speeds: $("speeds"),
  fastForward: $("fast-forward"),
  showTrue: $("show-true"),
  showEstimated: $("show-estimated"),
  viewer: $("viewer"),
  measureCom: $("measure-com"),
  simplified: $("simplified"),
  hold: $("hold"),
  seed: $("seed"),
  run: $("run"),
  stop: $("stop"),
  scrub: $("scrub"),
  tickStart: $("tick-start"),
  tickEnd: $("tick-end"),
  frameReadout: $("frame-readout"),
  pause: $("pause"),
  transport: document.querySelectorAll(".tbtn[data-act]"),
  log: $("log"),
  runTitle: $("run-title"),
};

let experiments = [];
let selected = null;
let speedName = "x1";
let speedValue = 1;
let mode = "execution";
let dragging = false;

const IDLE = { running: false, frames: 0, index: null, paused: false, playback: "stopped", holding: false, activity: "" };

let live = { ...IDLE };

/* -- talking to the server ------------------------------------------------------- */

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

/* Live switches only mean something to a run that is already going. */
function pushSettings() {
  if (!live.running) return;
  post("/api/control", {
    speed: speedValue,
    fast_forward: ui.fastForward.checked,
    show_true_com: ui.showTrue.checked,
    show_estimated_com: ui.showEstimated.checked,
  }).catch(() => {});
}

function command(message) {
  if (!live.running) return;
  post("/api/control", message).catch(() => {});
}

/* -- building the page ------------------------------------------------------------ */

function renderExperiments() {
  ui.experimentCount.textContent = `${experiments.length} niveles`;
  const nodes = [];
  for (const source of ["table", "conveyor", "truck"]) {
    const heading = document.createElement("h3");
    heading.className = "group-title";
    heading.textContent = { table: "Mesa", conveyor: "Cinta", truck: "Camión" }[source];
    nodes.push(heading);
    for (const item of experiments.filter((entry) => entry.source === source)) {
      const card = document.createElement("label");
      card.className = "card";
      card.dataset.key = item.key;

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "experiment";
      radio.value = item.key;
      radio.addEventListener("change", () => selectExperiment(item.key));

      const title = document.createElement("h3");
      title.textContent = item.title;

      const text = document.createElement("p");
      text.textContent = item.description;

      const tags = document.createElement("div");
      tags.className = "tags";
      for (const [label, cls] of tagsFor(item)) {
        const tag = document.createElement("span");
        tag.className = `tag ${cls}`;
        tag.textContent = label;
        tags.append(tag);
      }

      card.append(radio, title, text, tags);
      nodes.push(card);
    }
  }
  ui.experiments.replaceChildren(...nodes);
}

function tagsFor(item) {
  const tags = [];
  if (item.watchable) tags.push(["ventana 3D", "tag-3d"]);
  if (item.usesRobot) tags.push(["brazo", "tag-robot"]);
  if (item.source === "table") tags.push(["mesa", ""]);
  if (item.source === "conveyor") tags.push(["cinta", ""]);
  if (item.source === "truck") tags.push(["cami\u00f3n", "tag-truck"]);
  if (item.instantPlace) tags.push(["colocado instant\u00e1neo", ""]);
  if (item.shake) tags.push(["sacudidas", "tag-shake"]);
  if (item.kind === "plan") tags.push(["sin f\u00edsica", ""]);
  if (item.kind === "benchmark" || item.kind === "com-benchmark") tags.push(["benchmark", ""]);
  return tags;
}

function renderSpeeds(presets) {
  ui.speeds.replaceChildren(
    ...presets.map(([name, value]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = SPEED_LABELS[name] || name;
      button.dataset.name = name;
      button.addEventListener("click", () => {
        speedName = name;
        speedValue = value;
        refresh();
        pushSettings();
      });
      return button;
    }),
  );
}

function selectExperiment(key) {
  selected = experiments.find((item) => item.key === key) || experiments[0];
  refresh();
}

/* -- running ---------------------------------------------------------------------- */

async function start() {
  if (!selected || live.running) return;
  const seed = ui.seed.value.trim();
  try {
    await post("/api/run", {
      experiment: selected.key,
      mode,
      speed: speedValue,
      fast_forward: ui.fastForward.checked,
      show_true_com: ui.showTrue.checked,
      show_estimated_com: ui.showEstimated.checked,
      viewer: ui.viewer.checked,
      hold_at_end: ui.hold.checked,
      simplified_graphics: ui.simplified.checked,
      measure_com: ui.measureCom.checked,
      seed: seed === "" ? null : Number(seed),
    });
    live.running = true;
    refresh();
  } catch (error) {
    appendLine({ text: error.message, tone: "bad" });
  }
}

function stop() {
  if (!live.running) return;
  if (live.holding) {
    command({ command: "release" });
  } else {
    post("/api/stop").catch(() => {});
  }
  setStatus("Deteniendo\u2026", "pause");
}

/* -- the event stream -------------------------------------------------------------- */

function connect() {
  const source = new EventSource("/api/events");

  source.addEventListener("open", () => ui.link.classList.add("on"));
  source.addEventListener("error", () => ui.link.classList.remove("on"));

  source.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    switch (message.kind) {
      case "hello":
        live = { ...live, ...message.state, running: message.running };
        mode = message.mode || mode;
        ui.runTitle.textContent = message.title || "";
        setLog(message.log || []);
        break;
      case "started":
        live.running = true;
        mode = message.mode || mode;
        ui.runTitle.textContent = message.title || "";
        setLog([{ text: `\u25b6 ${message.title}`, tone: "head" }]);
        break;
      case "state":
        live = { ...live, ...message, running: true };
        break;
      case "log":
        appendLine({ text: message.text, tone: "note" });
        break;
      case "finished":
        (message.lines || []).forEach((text, index) =>
          appendLine({ text, tone: index === 0 ? (message.ok ? "ok" : "bad") : "note" }),
        );
        break;
      case "closed":
        live = { ...IDLE };
        break;
    }
    refresh();
  });
}

/* -- the log ------------------------------------------------------------------------ */

function setLog(lines) {
  ui.log.replaceChildren();
  if (!lines.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Elige un experimento y pulsa Ejecutar.";
    ui.log.append(empty);
    return;
  }
  lines.forEach(appendLine);
}

function appendLine(line) {
  ui.log.querySelector(".empty")?.remove();
  const paragraph = document.createElement("p");
  paragraph.className = line.tone || "note";
  paragraph.textContent = line.text;
  ui.log.append(paragraph);
  ui.log.scrollTop = ui.log.scrollHeight;
}

/* -- painting ------------------------------------------------------------------------ */

function refresh() {
  const busy = live.running;
  const recording = busy && live.frames > 0;

  ui.main.classList.toggle("busy", busy);
  for (const card of ui.experiments.querySelectorAll(".card")) {
    const on = selected && card.dataset.key === selected.key;
    card.classList.toggle("on", Boolean(on));
    card.querySelector("input").checked = Boolean(on);
    card.querySelector("input").disabled = busy;
  }
  for (const button of ui.speeds.children) {
    button.classList.toggle("on", button.dataset.name === speedName);
  }
  for (const button of ui.modes.children) {
    button.classList.toggle("on", button.dataset.mode === mode);
    button.disabled = busy;
  }
  ui.modeBadge.textContent = mode === "execution" ? "EJECUCIÓN" : "DEPURACIÓN · SIN TELEMETRÍA";
  ui.modeBadge.dataset.mode = mode;
  ui.modeNote.textContent = mode === "execution"
    ? "Lanza scripts/palletize.py y publica telemetría si hay credenciales."
    : "Usa el runner local de tools/stable_pallet. No abre episodios ni sube telemetría.";

  ui.viewer.disabled = busy || !selected?.watchable;
  ui.measureCom.disabled = busy || !selected?.usesRobot;
  ui.simplified.disabled = busy;
  ui.hold.disabled = busy;
  ui.seed.disabled = busy;
  if (selected && !selected.watchable) ui.viewer.checked = false;

  ui.run.disabled = busy;
  ui.stop.disabled = !busy;
  ui.stop.querySelector("span").textContent = live.holding ? "Cerrar visor" : "Detener";

  for (const button of ui.transport) button.disabled = !recording;
  ui.pause.disabled = !busy || live.holding;
  ui.pause.classList.toggle("on", live.paused);
  ui.pause.querySelector("span").textContent = live.paused ? "Reanudar" : "Pausa";
  ui.pause.querySelector("use").setAttribute("href", live.paused ? "#i-play" : "#i-pause");

  refreshTimeline();
  setStatus(...statusText());
}

function refreshTimeline() {
  const count = live.frames;
  if (!count) {
    ui.frameReadout.textContent = "sin grabaci\u00f3n";
    ui.scrub.max = 0;
    ui.scrub.value = 0;
    ui.scrub.disabled = true;
    ui.scrub.style.setProperty("--fill", "0%");
    ui.tickStart.textContent = ui.tickEnd.textContent = seconds(0);
    return;
  }
  const shown = live.index === null || live.index === undefined ? count - 1 : live.index;
  ui.scrub.max = Math.max(0, count - 1);
  ui.scrub.disabled = !live.running;
  if (!dragging) ui.scrub.value = shown;
  ui.scrub.style.setProperty("--fill", `${count > 1 ? (shown / (count - 1)) * 100 : 100}%`);
  ui.tickStart.textContent = seconds(0);
  ui.tickEnd.textContent = seconds((count - 1) * FRAME_SECONDS);
  ui.frameReadout.textContent = `fotograma ${shown + 1}/${count} \u00b7 ${seconds(shown * FRAME_SECONDS)} simulados`;
}

function seconds(value) {
  return `${value.toLocaleString("es-ES", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} s`;
}

function statusText() {
  if (!live.running) return ["Listo", ""];
  if (live.holding) return ["Terminado \u00b7 visor abierto para revisar", "hold"];
  /* The cell does not step while the planner thinks, so the window holds its last
     frame. Saying so is what separates slow from broken. */
  if (live.activity) return [live.activity, "work"];
  if (live.playback !== "stopped") {
    return [`Revisando la grabaci\u00f3n hacia ${live.playback === "forward" ? "delante" : "atr\u00e1s"}`, "pause"];
  }
  if (live.paused) return [`En pausa${live.index !== null ? " \u00b7 revisando la grabaci\u00f3n" : ""}`, "pause"];
  const motion = ui.fastForward.checked ? "fast-forward" : "trayectorias completas";
  return [`${mode === "execution" ? "Ejecución" : "Depuración"} \u00b7 ${SPEED_LABELS[speedName]} \u00b7 ${motion}`, "run"];
}

function setStatus(text, tone) {
  ui.status.textContent = text;
  if (tone !== undefined) ui.status.dataset.tone = tone;
}

/* -- wiring -------------------------------------------------------------------------- */

for (const input of [ui.fastForward, ui.showTrue, ui.showEstimated]) {
  input.addEventListener("change", () => {
    refresh();
    pushSettings();
  });
}

ui.run.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
ui.pause.addEventListener("click", () => command({ command: live.paused ? "resume" : "pause" }));

for (const button of ui.modes.children) {
  button.addEventListener("click", () => {
    if (live.running) return;
    mode = button.dataset.mode;
    refresh();
  });
}

for (const button of ui.transport) {
  button.addEventListener("click", () => {
    switch (button.dataset.act) {
      case "start": return command({ command: "scrub", index: 0 });
      case "end": return command({ command: "scrub", index: Math.max(0, live.frames - 1) });
      case "prev": return command({ command: "step", delta: -1 });
      case "next": return command({ command: "step", delta: 1 });
      case "backward": return command({ command: "play", direction: "backward" });
      case "forward": return command({ command: "play", direction: "forward" });
    }
  });
}

/* Holding the knob has to win over the reports arriving ten times a second. */
ui.scrub.addEventListener("pointerdown", () => { dragging = true; });
ui.scrub.addEventListener("pointerup", () => { dragging = false; });
ui.scrub.addEventListener("input", () => {
  ui.scrub.style.setProperty("--fill", `${(ui.scrub.value / Math.max(1, ui.scrub.max)) * 100}%`);
  command({ command: "scrub", index: Number(ui.scrub.value) });
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  if (event.code === "Space") { event.preventDefault(); ui.pause.click(); }
  if (event.code === "ArrowLeft") command({ command: "step", delta: -1 });
  if (event.code === "ArrowRight") command({ command: "step", delta: 1 });
});

/* -- start ---------------------------------------------------------------------------- */

fetch("/api/experiments")
  .then((response) => response.json())
  .then((data) => {
    experiments = data.experiments;
    renderSpeeds(data.speeds);
    renderExperiments();
    selectExperiment(experiments[0].key);
    connect();
  });
