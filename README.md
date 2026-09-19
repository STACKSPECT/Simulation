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
| `src/vision/gauge.py` | 🟡 Stub | `WristGauge.measure()` raises. Approach undecided. |
| `src/vision/oracle.py` | ⬜ Docstring only | `OracleDetector` / `OracleGauge` sketched in comments; ~10 lines each once `cell/scene.py` exists. |
| `src/planner/__init__.py` | ⬜ Docstring only | Package marker. |
| `src/planner/heightmap.py` | 🟡 Stub | `measure()` raises. |
| `src/planner/heuristic.py` | 🟡 Stub | `ScorePlanner.choose()` raises; `__init__` stores the config. |
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

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_KEY=
```

`SUPABASE_SERVICE_KEY` bypasses RLS and is the only one that can write — **Python side
only, never in a frontend.** Without these variables the SDK does not complain: it treats
"no credentials" as its normal mode and writes to disk only. That is precisely why the
entry point is required to always print which mode it is running in.

## Development

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full workflow. In short:

- **`dev` is the integration branch.** Branch from it, open pull requests against it.
  `main` is synced from `dev`.
- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor(scope):`, `chore:`), subject
  lines in Spanish.
- **Documentation is English; code, comments and configs are Spanish.** Do not translate
  the Spanish comments — they carry measured values and their reasoning.
- There is no CI. The verification ladder in `AGENTS.md` §9 takes its place, ordered
  cheapest-first. None of its six rungs can be run today.
- **The tests use no framework on purpose** — plain asserts run as a script
  (`python tests/test_pallet.py`), not pytest. Do not add one.

**Read [AGENTS.md](AGENTS.md) in full before writing code.** It is in Spanish and it is
where the project's real knowledge lives: the closed vocabularies, the hard rules, and
the traps that each cost a lost run in the predecessor repo.

## Intended usage

> **None of this works yet.** `scripts/palletize.py` raises `NotImplementedError`, and
> `tests/test_pallet.py` and `src/measure.py` contain no code. This section documents the
> interface the skeleton is being built towards, so that the design is reviewable — not
> commands you can run today.

```bash
python scripts/palletize.py --viewer            # watch it
python scripts/palletize.py -n 3                # 3 episodes; UPLOADS by default
python scripts/palletize.py -n 3 --no-telemetry # no upload, disk only
python tests/test_pallet.py                     # checks, no simulator, no network
python -m src.measure                           # the measurement, with its asserts
```

Plus three flags that select stub versus real implementation, which is the only place
that choice is made — and therefore the only place that can compute the run's `oracle`
flag:

```python
detector = OracleDetector() if args.oracle_vision else CameraDetector()
gauge    = OracleGauge()    if args.oracle_gauge  else WristGauge()
planner  = GridPlanner(cfg) if args.naive_planner else ScorePlanner(cfg)
oracle   = any([args.oracle_vision, args.oracle_gauge, args.naive_planner])
```

Three behaviours the entry point is required to have, each of which cost a lost run
before:

- **It prints which telemetry mode it is in.** A misplaced `.env` swallows uploads
  silently, and there is no other way to find out.
- **An open episode is always closed**, in a `finally`. A Ctrl-C leaves the row in
  `running` forever, and the Live screen picks the first episode in that state *without
  ordering* — one orphan pins it there indefinitely.
- **It passes `config=run_config(scene)`** including `pallet_size_m`. Without it the UI
  assumes a 1200×800 europallet and every dimension comes out wrong by the scale factor.

## The closed vocabularies

Three vocabularies from the platform contract are closed. Inventing a value **does not
raise an error where you write it** — it fails later and silently, and usually takes the
rest of the run's uploads with it. They are repeated here because this is the kind of
thing a README should warn about; the full reasoning is in `AGENTS.md` §5.

| | Allowed values |
|---|---|
| `events.kind` | `perceive`, `plan`, `pick`, `place`, `settle`, `fail` |
| `snapshots.view` | `top`, `side`, `iso`, `camera` |
| `failure` | `no_detection`, `ik_unreachable`, `collision`, `grasp_slip`, `wrong_placement`, `timeout`, `stack_collapse`, `overhang_violation` |

- **There is no event kind for the conveyor.** It is a closed CHECK of six values and
  none is its. Conveyor state rides in the `perceive` payload, where extra keys are
  harmless, or it does not travel. Invent a kind and the database rejects the row with a
  400 the SDK swallows — **uploads stay off for the rest of the run**.
- **A wrong view name fails after the upload.** The PNG lands in Storage and *then* the
  row is rejected with a `23514`: orphaned photo, trace with no image. The previous repo
  hit this with `front`.
- **A wrong failure cause raises `ValueError`** from `EpisodeResult`, deliberately.
  Adding one means changes in three places in Platform — ask the backend owner.
- A conveyor jam is **`timeout`**, not a new cause. If the station stays empty and the
  wait expires, that is `timeout`; if it delivers but perception sees nothing, that is
  `no_detection`.

And `task` must be `"palletizing"`, which the SDK validates too.

## Open decisions

### ⚠️ The robot arm has not been chosen — and this blocks the project

`configs/scene.yaml` has `robot.model` **blank**. It is meant to point at a `scene.xml`
inside `third_party/mujoco_menagerie/<arm>/`, and changing arms is changing that one
path — *and re-measuring everything else*.

**Every calibration figure the project inherited is a Franka Panda measurement.** They
come from the predecessor's `configs/pallet.yaml` header, which is a lab notebook, and
they are all properties of the Panda's gripper, not of this project:

| Value | Panda measurement | Why it is not portable |
|---|---|---|
| `cartesian_speed` | 0.0625 m/s | A quarter of nominal. Above it the box slips in the gripper and lands 15–20 mm off. It is the segment's acceleration, not the mass (tested 0.03–0.13 kg, same slip). |
| clearance between boxes | 23 mm | On the gripper's **closing axis**. Imposed by the finger — 10.5 mm thick plus what it opens on release — not by the box. Also the pallet's occupancy ceiling (72–76 %). |
| `drop_clearance` | 8 mm | The bottom of a measured curve: 20 mm → 5.1 mm error, 8 mm → 1.9 mm, 2 mm → 3.9 mm. |
| `release_margin` | 3 mm per side | The gripper does not open fully to release; it opens fully up high and away from the stack. |

Also arm-dependent and blank in `configs/scene.yaml`: `tcp_offset`,
`gripper_max_opening`, `gripper_actuator`, `observe_qpos`, and the IK convergence
tolerances — which must sit **above** the servo's measured steady-state droop under load,
or `move_to` never converges and everything gets reported as `ik_unreachable`.

**These get re-measured, not copied.** The measurement procedure is in the predecessor's
`configs/pallet.yaml` header, along with the IK reach sweep that has to be repeated.

### Undecided, but not blocking

- **The perception approach** (`src/vision/detect.py`): rendered RGB-D with classical
  segmentation, a segmentation buffer degraded with noise, or a trained model.
- **The gauging approach** (`src/vision/gauge.py`): a dedicated measuring station the arm
  carries the package to, or in-flight measurement from the robot's own sensors. Also
  which quantities are genuinely measured versus assumed — mass could come from wrist
  torque or from the delivery note; both are defensible, not knowing which is not.

## Dependencies and licences

Audited with `pip-licenses` in a clean virtual environment built from `requirements.txt`,
not assumed.

### Direct dependencies

| Package | Version | Licence | Why it is here |
|---|---|---|---|
| [mujoco](https://github.com/google-deepmind/mujoco) | 3.13.0 | Apache-2.0 | The physics. |
| [mink](https://github.com/kevinzakka/mink) | 1.3.0 | Apache-2.0 | Cartesian IK for the arm. |
| [qpsolvers](https://github.com/qpsolvers/qpsolvers) | 4.13.0 | **LGPL-3.0** | mink solves IK through it and ships no solver of its own. See the note below. |
| [daqp](https://github.com/darnstrom/daqp) | 0.9.1 | MIT | The QP backend qpsolvers actually calls. |
| [numpy](https://numpy.org) | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Everywhere. |
| [scipy](https://scipy.org/) | 1.18.1 | BSD-3-Clause | Geometry for the measurement module. |
| [PyYAML](https://pyyaml.org/) | 6.0.3 | MIT | Reads `configs/`. |
| [imageio](https://github.com/imageio/imageio) | 2.37.3 | BSD-2-Clause | Writes the pallet PNGs. |

Pinned to `==` and reconciled against the set `Guionized-simulation` had already verified
works together. Where the two repos disagreed, the predecessor's proven version wins.

### Transitive dependencies

All permissive: `absl-py` (Apache-2.0), `etils` (Apache-2.0), `glfw` (MIT), `pillow`
(MIT-CMU), `PyOpenGL` (BSD), `fsspec` (BSD-3-Clause), `typing_extensions` (PSF-2.0),
`zipp` (MIT).

### On `qpsolvers` and the LGPL

**`qpsolvers` is the only copyleft-licensed dependency in the tree** (LGPL-3.0). Stating
it explicitly rather than staying quiet about it:

It is used **as a library, imported at runtime from a virtual environment, and no part of
it is redistributed inside this repository**. The LGPL's copyleft attaches to the library
and to derivative works of it, not to software that merely uses it across that boundary;
users remain free to replace their installed copy. **It therefore does not force a
licence change here**, and this project's MIT licence stands.

Two conditions that come with that, worth knowing before anyone changes the packaging:

- If this project is ever **redistributed as a bundle that vendors or statically embeds**
  `qpsolvers` (a frozen binary, a container image shipped as a product, a wheel that
  includes it), the LGPL's relinking and notice obligations apply to that bundle.
- Modifying `qpsolvers` itself means releasing those modifications under the LGPL.

Neither applies to the repository as it stands.

### Not installed from `requirements.txt`

| Component | Source | Licence |
|---|---|---|
| **`theker_telemetry`** | [`STACKSPECT/Platform`](https://github.com/STACKSPECT/Platform), branch `dev`, installed editable by `scripts/setup.sh`. Deliberately not in `requirements.txt`: it is the contract of whoever stores the data. | Same project, same organisation. |
| **[`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)** | Cloned into `third_party/` by `scripts/setup.sh` at install time. Listed in `.gitignore` — **nothing from it is redistributed here.** | The repository as a whole is **Apache-2.0**, © 2022 DeepMind Technologies Limited. **Individual model directories carry their own terms** (Apache-2.0, BSD-3-Clause or MIT, depending on the model) — consult the `LICENSE` file inside the model subdirectory you end up selecting for `robot.model`. Credit: MuJoCo Menagerie, Google DeepMind. |

## Licence

[MIT](LICENSE) © 2026 STACKSPECT.

The dependency audit above found no licence incompatible with releasing this project
under MIT.
