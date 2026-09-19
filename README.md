# Simulation

Palletizing in MuJoCo: a conveyor stops, an arm picks a package off it, measures the
package **in hand**, decides where it goes, and stacks it on a pallet. The physics is
real and the metrics are measured, not estimated. Everything streams live to the
[Platform](https://github.com/STACKSPECT/Platform) observability project.

> ## ⚠️ Status: skeleton. It does not run yet.
>
> The contracts are fixed and the architecture document is written. **One Python module
> is implemented** (`src/contracts.py`) and **one script works** (`scripts/setup.sh`).
> Every other module is a docstring describing what goes in it, and its functions raise
> `NotImplementedError`.
>
> There is no working `palletize.py`, no test suite, no measurement module. The
> [per-file status table](#per-file-status) below says exactly what exists. Please read
> it before you try to run anything — the usage section describes the **intended**
> interface, not a working one.

---

## Contents

- [What this is](#what-this-is)
- [How it differs from its predecessor](#how-it-differs-from-its-predecessor)
- [Architecture](#architecture)
- [Per-file status](#per-file-status)
- [Requirements](#requirements)
- [Installation](#installation)
- [Development](#development)
- [Intended usage](#intended-usage)
- [The closed vocabularies](#the-closed-vocabularies)
- [Open decisions](#open-decisions)
- [Dependencies and licences](#dependencies-and-licences)
- [Licence](#licence)

---

## What this is

A palletizing cell simulated in MuJoCo, built so that the interesting parts are actually
decided rather than scripted. The cycle of one package — which is also the order the
modules are called in:

```
conveyor advances and STOPS      cell.conveyor.present()
  -> it is seen                  vision.detect.observe()      -> Observation
  -> it is picked                cell.arm                     (execution)
  -> it is measured in hand      vision.gauge.measure()       -> PackageSpec
  -> a slot is chosen            planner.heuristic.choose()   -> PlacementPlan
  -> it is placed                cell.arm                     (execution)
  -> the result is measured      measure.measure_placement()  -> Placement, PalletState
  -> it is reported              telemetry.RunLogSink         (rows, live)
```

Note the order: **measurement happens AFTER picking and BEFORE planning.** That is not a
design flourish — it is what forces the planner to be incremental. It cannot precompute
the whole pallet, because it does not know what the next package looks like until the arm
is holding it.

## How it differs from its predecessor

[`STACKSPECT/Guionized-simulation`](https://github.com/STACKSPECT/Guionized-simulation)
("scripted simulation") did the same job with every box's slot written out in a YAML
file. It existed to pin down the contract with the platform and walk the whole thing
end-to-end before the real system existed. **This repository is the real system.** The
difference is not cosmetic:

| | scripted (predecessor) | here |
|---|---|---|
| where each box goes | written in `configs/pallet.yaml` | decided by a heuristic from what it sees and measures |
| what is on the conveyor | nothing — boxes wait pre-placed on the table | reported by perception, which can be wrong |
| package dimensions | read from the catalogue | measured with the package already in hand |
| run's `oracle` flag | always `true` | **`false`** once no stub is left in the loop |
| `no_detection` failure | unreachable | reachable |

## Architecture

Three layers plus two cross-cutting concerns. Each folder is a boundary: you work inside
one without opening the others.

| Layer | Folder | What it does |
|---|---|---|
| **Vision** | `src/vision/` | Sees the package stopped on the conveyor (`detect.py`) and **measures** it once in hand: dimensions, mass, centre of gravity (`gauge.py`) |
| **Planning** | `src/planner/` | Height map of the pallet (`heightmap.py`) and the **scoring heuristic** that picks the slot (`heuristic.py`) |
| **Execution** | `src/cell/` | MuJoCo: scene, arm, **conveyor** that starts and stops, cameras |
| Measurement | `src/measure.py` | The result: error, support, overhang, pallet CoG, stability margin |
| Traceability | `src/telemetry.py` | The **only** boundary with the platform |

`src/episode.py` stitches them together — it asks, it does not compute. And they all
speak the same language:

### The shared language lives in `src/contracts.py`

**This is the one file all four modules read**, and the only one that has to be agreed
before anything else is touched. It knows nothing about the platform: the names are the
domain's, not any column's. Units are SI throughout — metres, kilograms, seconds,
radians. The single exception in the project is event `payload`s, which go in mm and
degrees, and that conversion happens at the telemetry boundary and nowhere else.

| Type | Produced by | Consumed by, and what it carries |
|---|---|---|
| `Observation` | `vision/detect.py` | `episode.py` — id, pose on the conveyor, approximate dims, `confidence` |
| `PackageSpec` | `vision/gauge.py` | `planner`, `telemetry` — `dims_m`, `mass_kg`, `cog_offset_m`, `grasp_width` |
| `Heightmap` | `planner/heightmap.py` | `planner/heuristic.py` — grid of heights above the pallet deck |
| `PlacementPlan` | `planner/heuristic.py` | `episode.py` — target pose, `layer`, `slot`, `score`, predicted support |
| `Placement`, `PalletState` | `measure.py` | `telemetry.py` — what was measured |

Plus four `Protocol`s — `Detector`, `Gauge`, `Planner`, `Sink` — which are what lets the
four modules be built in parallel without blocking on each other.

**The rule that makes that work: each module ships its oracle stub BEFORE its real
implementation.** The stub reads ground truth straight out of the scene and returns a
valid contract object. With all four stubs in place the whole loop runs green on day one,
and swapping one for the real thing is a one-line change in `scripts/palletize.py`. The
stubs live next to what they replace, clearly named: `vision/oracle.py`,
`planner/naive.py`.

A run's `oracle` flag is `any(stub in use)`. Marking it wrong invalidates exactly the
comparison that justifies the work, because the UI does not compare an oracle run against
a non-oracle one.

## Per-file status

Verified file by file against the source on this branch, not from memory.

> **Stale below the planner rows.** This table was written for the skeleton and was not
> refreshed when the UR10e cell landed on `dev`: `episode.py`, `measure.py`,
> `telemetry.py`, `cell/*` and `scripts/palletize.py` are all implemented today, whatever
> the rows say. Only the rows touched by the placement heuristic have been brought up to
> date here. Fixing the rest is a separate pass.

**Implemented** means there is working code. **Stub** means the functions or classes are
declared with the right signature but raise `NotImplementedError`. **Docstring only**
means the file contains its specification and nothing executable — the work is to port or
write it.

### Python

| File | Status | Notes |
|---|---|---|
| `src/contracts.py` | ✅ **Implemented** | The only implemented module. 4 dataclasses, 4 `Protocol`s, derived properties (`grasp_width`, `footprint_area`, `Heightmap.top`). Imports and works. |
| `src/episode.py` | 🟡 Stub | `run_episode()` raises. Skeleton to be ported from the predecessor. |
| `src/measure.py` | ⬜ Docstring only | No code. To be ported wholesale, with one change: `pallet_state` must accumulate CoG using `cog_offset_m`. |
| `src/telemetry.py` | ⬜ Docstring only | No code. To be ported; its row functions are the verified contract. |
| `src/cell/__init__.py` | ⬜ Docstring only | To be ported as-is (`TCP_SITE`, `add_table`, `tcp_frame`, `lookat_quat`). |
| `src/cell/arm.py` | ⬜ Docstring only | To be ported as-is (`ArmController`, mink IK). |
| `src/cell/scene.py` | ⬜ Docstring only | To be ported with changes: boxes no longer come from a script. |
| `src/cell/render.py` | ⬜ Docstring only | To be ported (`render(scene, view)`). |
| `src/cell/conveyor.py` | 🟡 Stub | `present()` and `release()` raise. **Written from scratch** — no predecessor to port from. |
| `src/vision/__init__.py` | ⬜ Docstring only | Package marker. |
| `src/vision/detect.py` | 🟡 Stub | `CameraDetector.observe()` raises. Approach undecided (rendered RGB-D, degraded segmentation buffer, or a trained model). |
| `src/vision/gauge.py` | ✅ **Implemented** | `WristGauge`, wrist-load based. |
| `src/vision/surface.py` | ✅ **Implemented** | Depth → world points → pallet grid → fused height map. **numpy only**, no MuJoCo: that is what makes the fusion testable without a simulator. Ported from `pallet_perception`. |
| `src/vision/depth.py` | ✅ **Implemented** | The MuJoCo side: renders depth from a named camera and reads its pose and intrinsics. The only perception file that touches the simulator. |
| `placing/` (9 files) | ✅ **Vendored, do not edit** | The placement heuristic: enumerate every discrete pose, hard-filter, score with 14 terms. numpy and nothing else; `python -m placing` runs 15 checks, the first of which asserts that boundary. Copied whole from `placing-algo`. |
| `src/vision/oracle.py` | ⬜ Docstring only | `OracleDetector` / `OracleGauge` sketched in comments; ~10 lines each once `cell/scene.py` exists. |
| `src/planner/__init__.py` | ⬜ Docstring only | Package marker. |
| `src/planner/heightmap.py` | ✅ **Implemented** | `measure()` fuses three depth cameras into the pallet grid; `measure_ground_truth()` is the oracle, moved here from `naive.py`, and stamps the table where it intrudes over the deck. |
| `src/planner/heuristic.py` | ✅ **Implemented** | `ScorePlanner`, the adapter over `placing/`. ~200 lines, of which the interesting part is four unit translations that fail silently. The default planner; `--beam-planner` and `--naive-planner` opt out. |
| `src/planner/naive.py` | ⬜ Docstring only | `GridPlanner` sketched in a comment; ~20 lines. |
| `scripts/palletize.py` | 🟡 Stub | `main()` raises. This is the entry point and the only place that picks stub vs. real. |
| `tests/test_pallet.py` | ⬜ Docstring only | **No tests exist.** The file lists what to port and what to add. |

### Everything else

| File | Status | Notes |
|---|---|---|
| `scripts/setup.sh` | ✅ **Implemented** | Works end to end: venv, deps, editable SDK from Platform with an assertion that it is the `dev` branch, and clones `mujoco_menagerie`. The only executable path in the repo. |
| `requirements.txt` | ✅ Complete | All pinned to `==`, reconciled with the predecessor's verified set. |
| `.env.example` | ✅ Complete | The three Supabase variables, with the warning about `SERVICE_KEY`. |
| `configs/scene.yaml` | 🟡 Keys only | Parses as valid YAML; **every value is blank**, including `robot.model`. |
| `configs/pallet.yaml` | 🟡 Keys only | Parses; values blank, `packages` commented out as an example. The header documents what to port and re-measure. |
| `AGENTS.md` | ✅ Complete | 19 KB, Spanish. The project's real knowledge. |

## Requirements

- **Python 3.11+** (developed and verified on 3.12).
- **Linux or macOS.** `scripts/setup.sh` is bash and assumes `python3` and `git` on PATH.
- **`git`**, to clone the arm model at install time.
- **A checkout of [`STACKSPECT/Platform`](https://github.com/STACKSPECT/Platform)** — see
  below. This is a hard requirement, not an optional integration.
- **A GPU is not required**, but rendering the pallet views needs a working OpenGL
  context.

### The hard cross-repo dependency

**This repository does not stand alone.** The `theker_telemetry` SDK is deliberately not
vendored here: it is the contract of what the platform stores, and whoever stores the
data defines its shape. So:

> You need the `Platform` repository checked out **as a sibling directory**, on its
> **`dev` branch**, installed **editable**. Without it, nothing imports.

```
parent-directory/
├── Simulation/      <- this repo
└── Platform/        <- required sibling, on branch `dev`
```

It has to be `dev`. `main` does not have `pallet.py` — the source of `stability_margin`
and `support_polygon` — and does not have the episode lifecycle (`RunLog.begin` /
`end` / `snapshot`). **That failure does not appear at install time, it appears at run
time**, which is why `scripts/setup.sh` asserts on it and stops.

If your clone is somewhere else, pass `PLATFORM=/path/to/Platform`.

## Installation

```bash
git clone https://github.com/STACKSPECT/Simulation
git clone -b dev https://github.com/STACKSPECT/Platform      # the required sibling

cd Simulation
bash scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` does four things, and it is the one path in this repo that is known to
work today:

1. Creates `.venv` and installs `requirements.txt`.
2. Installs `theker_telemetry` editable from `$PLATFORM/backend` (default `../Platform`).
3. Asserts the SDK exposes `RunLog.begin`, `end` and `snapshot`, plus `stability_margin`
   and `support_polygon` — i.e. that you are on Platform's `dev`. It exits if not.
4. Clones `mujoco_menagerie` into `third_party/` (ignored by git, never vendored).

On success it prints `SDK ok: ciclo de vida y geometría disponibles`.

**This installs the environment. It does not give you a runnable simulation** — see the
status table.

### Credentials

Copy `.env.example` to `.env`, in this repo or its parent directory:

[MIT](LICENSE) © 2026 STACKSPECT.

The dependency audit above found no licence incompatible with releasing this project
under MIT.
