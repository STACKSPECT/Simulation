# Simulation

Closed-loop palletizing in MuJoCo. A **Universal Robots UR10e** with an **OnRobot VGP20**
suction array takes packages from a table, a conveyor or a truck trailer, weighs each one
in the air, decides where it goes on the pallet, places it, and measures what actually
happened. Every run streams live to the
[Platform](https://github.com/STACKSPECT/Platform) observability project.

<div align="center">
  <img src="docs/img/cell-overview.png" alt="The simulated cell: the UR10e on its pedestal holds a cardboard box with the suction gripper, a conveyor loaded with boxes feeds it from the left, empty pallets wait on the right, yellow racking behind" width="900">
</div>

Nothing about the placement is scripted. The arm does not know what the next package
looks like until it is holding it, and the slot is chosen from what the cameras and the
wrist measured — which is the whole point of the project, and the reason the loop is
built in the order it is.

---

## Contents

- [What it does](#what-it-does)
- [The cell](#the-cell)
- [How a slot gets chosen](#how-a-slot-gets-chosen)
- [The control panel](#the-control-panel)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [What is not implemented yet](#what-is-not-implemented-yet)
- [Telemetry and the closed vocabularies](#telemetry-and-the-closed-vocabularies)
- [Dependencies](#dependencies)
- [Licence](#licence)

---

## What it does

One package, start to finish. This is also the order the modules are called in:

```
the source presents and STOPS   cell.conveyor.Supply.present()
  -> it is seen                 vision.detect.observe()      -> Observation
  -> it is picked               cell.arm                     (execution)
  -> it is weighed in hand      vision.gauge.measure()       -> PackageSpec
  -> a slot is chosen           planner.heuristic.choose()   -> PlacementPlan
  -> it is placed               cell.arm                     (execution)
  -> the result is measured     measure.measure_placement()  -> Placement, PalletState
  -> it is reported             telemetry.RunLogSink         (rows, live)
```

Note the order: **measurement happens after picking and before planning.** That is what
forces the planner to be incremental. It cannot precompute the pallet, because it does
not know the next package until the arm is holding it.

`src/episode.py` is the director — it asks, it does not compute — and every module speaks
the same language, defined once in `src/contracts.py`: `Observation`, `PackageSpec`,
`Heightmap`, `PlacementPlan`, `Placement`, `PalletState`, plus four `Protocol`s
(`Detector`, `Gauge`, `Planner`, `Sink`) that let the layers be swapped one at a time.

### How it differs from its predecessor

[`STACKSPECT/Guionized-simulation`](https://github.com/STACKSPECT/Guionized-simulation)
did the same job with every box's slot written out in a YAML file. It existed to pin down
the contract with the platform before the real system existed. **This repository is the
real system.**

| | scripted (predecessor) | here |
|---|---|---|
| where each box goes | written in `configs/pallet.yaml` | scored and chosen from what is measured |
| what the source delivers | nothing — boxes wait pre-placed | a table, a physical conveyor, or a loaded trailer |
| package dimensions and mass | read from the catalogue | measured with the package already in hand |
| pallet surface | assumed | fused from three depth cameras |
| `no_detection` failure | unreachable | reachable, once perception exists |

## The cell

| Layer | Folder | What it owns |
|---|---|---|
| **Vision** | `src/vision/` | `detect.py` sees what the source presents; `gauge.py` weighs it on the wrist; `depth.py` renders depth and `surface.py` fuses it into the pallet grid (NumPy only, no simulator) |
| **Planning** | `src/planner/` | `heightmap.py` measures the pallet surface; `heuristic.py` is the adapter onto `placing/`; `naive.py` holds the grid baseline and the beam search |
| **Execution** | `src/cell/` | MuJoCo: `scene.py` builds the MJCF cell, `arm.py` drives the UR10e and the vacuum, `conveyor.py` / `table.py` / `truck.py` are the three sources, `render.py` the four camera views |
| Measurement | `src/measure.py` | Placement error, support, overhang, pallet CoG, stability margin |
| Traceability | `src/telemetry.py` | The **only** boundary with the platform. No other module knows a column name |
| Heuristic | `placing/` | Vendored, NumPy-only scoring engine. Nine files, no simulator, no config loader |
| Demonstrator | `tools/` | The project `src/cell/` was carved out of, kept whole and runnable, plus the control panel |

### Robot and tool

Decided and calibrated: **UR10e on a pedestal, OnRobot VGP20 with 16 Ø40 mm cups.**
1300 mm reach, 12.5 kg payload; the tool weighs 2.55 kg, so the package ceiling is 8.5 kg.
The kinematics and inertias come from [MuJoCo
Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The pallet is a real europallet,
1.20 × 0.80 m, at scale 1.

Every calibrated number lives next to its measurement in
[`configs/scene.yaml`](configs/scene.yaml) and [`configs/pallet.yaml`](configs/pallet.yaml).
Do not eyeball them; if you change one, write down how you measured it.

### The three sources, and the nine levels

All three implement one contract, `Supply.present()` / `Supply.release()`:

- **Table** — packages staged ready, nothing moves.
- **Conveyor** — a physical belt that advances and stops. `present()` returns on *measured*
  rest, not after a fixed settle.
- **Truck** — the trailer presents its whole load at once and only ever offers the highest
  box, the one holding nothing up.

Three levels per source, declared in `configs/pallet.yaml` and selectable with `--level`:

| | N1 | N2 | N3 |
|---|---|---|---|
| **Table** | `11` one type, aligned | `12` mixed, rotated | `13` random, adversarial CoG |
| **Conveyor** | `21` one type, centred on the belt | `22` mixed, off-centre arrival | `23` random, variable spacing |
| **Truck** | `31` ordered columns | `32` with the loader's disorder | `33` random, adversarial CoG |

### Weighing in hand

Each package type carries a `cog_offset_m`: the centre of mass is *not* the geometric
centre. `vision/gauge.py` estimates it from a single plumb wrist reading — the two
horizontal components come out exact, the vertical one is anchored to the geometric
centre because nothing downstream reads it — and `--precise-com` sweeps several poses when
that is not enough. The method, its five preconditions and its failure modes are written
up in [`pesaje-en-el-sitio.md`](pesaje-en-el-sitio.md).

## How a slot gets chosen

The height map of the pallet is **measured, never tracked in a counter**: three depth
cameras (the overhead one plus two opposed diagonals) render depth, the points are
unprojected into the world, everything not facing upwards is discarded, and the rest is
rasterised onto the pallet footprint and fused by keeping the highest value per cell. A
cell no camera saw is marked unobserved — zero height is not the same as free deck.

With that map and the package in hand, `placing/` enumerates every discrete pose that
fits, hard-filters the infeasible ones and scores the rest as a weighted sum of
**fourteen terms**:

```
support_ratio  com_margin  lowness    void_fill   levelness  peak_penalty  lateral_proximity
edge_flush     seam_break  overhang   pallet_com  reachability  bridge      gap_waste
```

The score is a number in `[0, 1]` and the chosen slot is the maximum. The full breakdown
travels with the `plan` event, so "why did it put that box there?" is answerable without
re-running the episode. To change the behaviour you change the weights in
`configs/pallet.yaml: heuristic.weights` — a weight of 0 switches its term off — not the
code.

`placing/` is vendored and **must not be edited**. Its boundary is enforced by an
executable assert, not a convention: `python -m placing` runs fifteen checks and the first
one asserts that no heavy module has leaked into `sys.modules`. That boundary is what lets
the weights be tuned in milliseconds instead of minutes of physics.

<div align="center">
  <img src="docs/img/pallet-stacking.png" alt="The UR10e lowering a cardboard box onto a europallet already stacked two layers deep with green product, yellow racking alongside" width="900">
</div>

Two alternatives exist purely to be compared against it: `--naive-planner` (a grid
baseline, which counts as an oracle) and `--beam-planner` (the measured beam search from
`tools/`, which does not). The demonstrator's own benchmark puts the beam search at 100 %
strictly-stable stacks against first-fit's 50 %, with mean CoM off-centring down from
142 mm to 81 mm; any new heuristic is measured against those numbers.

When nothing scores, that is **not an exception**: it is a `wrong_placement`, and the
reason why travels in the `fail` event so "it does not fit" stays distinguishable from
"the arm can no longer reach".

## The control panel

`tools/` ships a browser panel for driving experiments. It is **stdlib only** — no
framework, no bundler, nothing fetched at start-up — and it talks to the cell over JSON
on a pipe. Nothing in `webapp.py` touches MuJoCo.

<div align="center">
  <img src="docs/img/control-panel.png" alt="The run control panel: a nine-level experiment picker on the left with playback transport and a log pane, mode, speed, centre-of-mass and start-up controls on the right, and a DEPURACIÓN · SIN TELEMETRÍA badge in the header" width="900">
</div>

```bash
cd tools
uv sync --extra dev --extra video
uv run stable-pallet dashboard          # prints: Panel en http://127.0.0.1:8000/
```

What it gives you:

- **Nine experiment cards**, read straight from `configs/pallet.yaml`, grouped by source.
  The panel keeps no level list of its own — add a level to the YAML and it appears.
- **Two modes, and it says which one it is in.** *Ejecución de verdad* runs
  `scripts/palletize.py` and publishes telemetry if credentials exist. *Depuración* runs
  the local runner in `tools/`, never opens an episode and never uploads a row — the
  header badge reads `DEPURACIÓN · SIN TELEMETRÍA` so a debugging session cannot be
  mistaken for a recorded one.
- **Speed** from ×0,5 to *máx*, changeable while the run is under way. Separately,
  **fast-forward** skips the trajectories — the arm jumps waypoint to waypoint — while
  still simulating the parts that decide the outcome: the release, the settle, the jolts.
- **Centres of mass**: *reales* (the ones MuJoCo integrates) against *calculados* (the
  ones the robot derived from the wrist), with the error drawn between them as a yellow
  line. That single overlay is what makes a bad weighing visible before it becomes a
  collapse.
- **Playback**: pause, scrub, step a frame either way, play backwards. Rewinding reviews
  what was recorded; the run waits where it was and continues from there.
- **Start-up options**: 3D window, weigh each box, simplified graphics, keep the viewer
  open at the end, and a seed override.

The panel is not the repository's entry point and deliberately cannot become one:
`scripts/palletize.py` is the only place that chooses real components against oracle
stubs, and therefore the only place that can compute a run's `oracle` flag.

## Requirements

- **Python 3.11+** (verified here on 3.12.3).
- **Linux or macOS.** `scripts/setup.sh` is bash and wants `python3` and `git` on PATH.
- **A working OpenGL context** for rendering. A GPU is not required; headless runs use
  EGL, which `scripts/palletize.py` selects for you unless `--viewer` is passed.
- **[`uv`](https://docs.astral.sh/uv/)**, only for `tools/` — the demonstrator and the
  control panel. The simulation itself does not need it.
- **A checkout of [`STACKSPECT/Platform`](https://github.com/STACKSPECT/Platform) on its
  `dev` branch.** This is a hard requirement, not an optional integration.

### The cross-repo dependency, and why it is `dev`

The `theker_telemetry` SDK is deliberately not vendored here: it is the contract of what
the platform stores, and whoever stores the data defines its shape. So Platform has to be
checked out and installed editable:

```
parent-directory/
├── Simulation/      <- this repo
└── Platform/        <- required sibling, on branch dev
```

It has to be `dev`. Platform's `main` ships only `core.py` and `schema.py`; `pallet.py` —
the source of `stability_margin` and `support_polygon` — is on `dev`, and so is the live
episode lifecycle (`RunLog.begin` / `end` / `snapshot`). **That failure does not appear at
install time, it appears at run time**, which is why `scripts/setup.sh` asserts on it and
stops. Point it elsewhere with `PLATFORM=/path/to/Platform`.

## Installation

```bash
git clone https://github.com/STACKSPECT/Simulation
git clone -b dev https://github.com/STACKSPECT/Platform      # the required sibling

cd Simulation
bash scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` does four things:

1. Creates `.venv` and installs `requirements.txt`.
2. Installs `theker_telemetry` editable from `$PLATFORM/backend` (default `../Platform`).
3. Asserts the SDK exposes `RunLog.begin`, `end` and `snapshot` plus `stability_margin`
   and `support_polygon` — i.e. that you are on Platform's `dev`. It exits if not.
4. Clones `mujoco_menagerie` into `third_party/` (git-ignored, never vendored).

On success it prints `SDK ok: ciclo de vida y geometría disponibles`.

### Credentials

Copy `.env.example` to `.env`, in this repo or in its parent directory:

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_KEY=
```

`SUPABASE_SERVICE_KEY` bypasses RLS and is the only one that can write — Python side only,
never a frontend. **With a `.env` present, telemetry is on by default**, and every run
prints which mode it is in. The SDK treats "no credentials" as its normal mode and writes
to disk silently, so that printed line is the only thing standing between you and a run
you thought was uploading.

## Usage

`scripts/palletize.py` is the entry point.

```bash
python scripts/palletize.py --viewer                 # watch one episode
python scripts/palletize.py -n 3                     # three episodes; UPLOADS by default
python scripts/palletize.py -n 3 --no-telemetry      # disk only
python scripts/palletize.py --source truck           # first level of that source
python scripts/palletize.py --level 23               # one specific level
```

```
$ python scripts/palletize.py -n 1 --no-telemetry --level 11
telemetría: solo disco (--no-telemetry)
modo: oráculo · ScorePlanner · mapa oráculo · mesa · N1 · un tipo, alineado · 1 episodio(s)
  std_m-00 std_m      L1 error   2.7 mm  apoyo 100% margen +124.9 mm  ok
  std_m-01 std_m      L1 error   3.4 mm  apoyo 100% margen +230.6 mm  ok
  std_m-02 std_m      L1 error   2.6 mm  apoyo  99% margen +272.1 mm  ok
  std_m-03 std_m      L2 error   2.6 mm  apoyo  88% margen +253.7 mm  ok
semilla 1: 4/4 · ÉXITO
1/1 episodios con éxito
disco: runs/20260920-010913-pallet
```

Swapping components in and out — the flags that decide whether a run is an oracle:

```bash
python scripts/palletize.py --no-oracle-gauge        # weigh on the wrist for real
python scripts/palletize.py --no-oracle-heightmap    # build the map from the cameras
python scripts/palletize.py --beam-planner           # the beam search, for comparison
python scripts/palletize.py --naive-planner          # the grid baseline
```

The startup line always names the planner and where the height map came from, because two
runs being compared have to be told apart by reading the output, not by remembering which
flags were typed.

Also useful: `--speed` (`0` = as fast as the machine manages), `--seed`, `--pause`,
`--show-com`, `--simplified-graphics`, `--video timelapse.mp4`, and `--protocol json`,
which is how the control panel drives it.

### What lands on disk

Disk is the source of truth and Supabase is a replica. Each run writes
`runs/<timestamp>-pallet/episodes.jsonl` — one line per episode, with its metrics and the
planner's full score breakdown — plus the top and side PNGs for each layer under
`runs/<timestamp>-pallet/<seed>/`. A network failure warns once, switches uploading off
and lets the episode finish.

## Development

Read **[`AGENTS.md`](AGENTS.md)** in full before writing any code. It is in Spanish, it is
the only place the project's real knowledge lives, and every hard rule in it cost a lost
run in the predecessor repository. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers branches,
the commit convention and the language rule (code and comments Spanish, outward-facing
docs English).

Work lands on `dev`. Branch from it, target it.

### The verification ladder

Ordered cheapest first on purpose — run it in this order and stop at the first failure.
All six rungs pass on this branch; the results below are what they printed here:

| | Command | What it proves | Result |
|---|---|---|---|
| 0 | `python -m placing` | The heuristic alone. The first check is the import boundary | `15 checks passed`, about a second |
| 1 | `python tests/test_pallet.py` | Row keys are columns, `seq` never repeats, the vocabularies hold, and the adapter's unit translations that otherwise fail silently | `20 comprobaciones pasadas` |
| 2 | `python -m src.measure` | CoG with out-of-tolerance boxes, margin against the support polygon | `ok measure.demo` |
| 3 | `python tests/test_cell.py` | Starts MuJoCo: all three sources compile, the cameras, the belt, the truck's unloading order, the IK envelope, the wrist gauge | `9 comprobaciones físicas pasadas` |
| 4 | `python scripts/palletize.py -n 1 --no-telemetry --level 21` | A whole episode to disk | `4/4 · ÉXITO` |
| 5 | `cd tools && uv run pytest` | The demonstrator survived being moved | `179 passed` in 67 s |

Then, with telemetry on: the run must print that it is active, the episode must appear
*in progress* in the UI within seconds, and the pallet must build package by package. The
SQL checks that catch the rest are in `AGENTS.md` §9.

### The tests use no framework, on purpose

`tests/test_pallet.py`, `tests/test_cell.py`, `python -m placing` and `python -m src.measure`
are plain asserts run as scripts. Do not add pytest, a `conftest.py` or fixtures to them —
the point is that they run in seconds with nothing installed beyond the dependencies, and
that anyone can read the file top to bottom. `tools/` is the exception: it is a separately
packaged project and it does use pytest.

### Boundaries that are not negotiable

- **Platform knowledge lives only in `src/telemetry.py`.** What crosses module boundaries
  are `contracts.py` objects, never rows.
- **Do not reimplement `stability_margin` or `support_polygon`.** They come from the SDK. A
  third copy would drift from the one the UI draws.
- **Do not let a module read the scene on its own.** If `heuristic.py` imports `mujoco`,
  something went wrong. The oracle stubs are the deliberate exception, which is why they
  live in separate, clearly named files.
- **Do not put anything inside `placing/`**, not even a convenience import.

## What is not implemented yet

The repository runs end to end today, on all three sources. Two things are honestly still
open, and the second follows from the first.

**Perception is not implemented.** `src/vision/detect.py::CameraDetector.observe()` is the
only `NotImplementedError` left in the codebase. The approach is undecided — rendered
RGB-D with classical segmentation, a degraded segmentation buffer, or a trained model —
and that decision is not this file's to make. `src/vision/oracle.py` covers the gap by
reading the truth out of the scene. Running `--no-oracle-vision` today raises:

```
NotImplementedError: percepción sin implementar: usa vision.oracle
```

**So every run is still flagged `oracle = true`.** A run's flag is `any(stub in use)`, and
with detection unavailable no combination of flags clears it. The gauge, the height map and
the planner can each be switched to their real implementation — and the height map
genuinely runs off the three cameras — but the flag stays `true` until perception lands.
It is computed in one place, `scripts/palletize.py`, and written by hand nowhere.

Two further honest notes from running this branch:

- **The camera height map is not yet at parity with the oracle.** On level 11, seed 1,
  `--oracle-heightmap` places 4/4; `--no-oracle-heightmap` places 3/4 and ends in
  `wrong_placement`. The path works; the numbers do not match yet. `allow_unobserved`
  therefore starts at `true`: camera coverage over an empty pallet measures 93.3 %, not the
  98 % that would make `false` reasonable.
- **The reachability term is switched off.** The adapter does not pass `robot_xy`, so
  `reachability` scores neutral and nothing is rejected for being out of reach — the reach
  figures inside `placing/` are a Panda's. Until that sweep is re-measured for the UR10e,
  the planner can pick a slot the arm cannot get to, which surfaces as `ik_unreachable`.

## Telemetry and the closed vocabularies

Rows are written **live** — `begin()` opens the episode, `event()` / `placement()` /
`pallet_state()` / `snapshot()` drop rows as they are measured, `end()` closes it with a
PATCH — which is the only reason the platform's Live screen is live. An open episode is
closed **always**, including on Ctrl-C: a single orphan left in `running` pins that screen
indefinitely.

Three vocabularies are closed CHECKs in the database. Inventing a value does not raise an
error where you write it — it fails later, silently, and usually takes the rest of the
run's uploads with it. All three are anchored against the schema by `tests/test_pallet.py`:

| Vocabulary | Allowed values | What an invented value does |
|---|---|---|
| `events.kind` | `perceive` `plan` `pick` `place` `settle` `fail` | Rejected with a 400 the SDK swallows, and uploads stay off for the rest of the run. There is deliberately **no kind for any of the sources** — their state rides in the `perceive` payload |
| `snapshots.view` | `top` `side` `iso` `camera` | The PNG uploads to Storage and *then* the row is rejected with a `23514`: an orphaned photo and a trace with no image |
| `failure` | `no_detection` `ik_unreachable` `collision` `grasp_slip` `wrong_placement` `timeout` `stack_collapse` `overhang_violation` | `EpisodeResult` raises `ValueError`, on purpose. Adding one means touching three places in Platform — ask the backend owner |

And the asymmetry worth memorising: **inside an event `payload` extra keys are harmless
and missing keys break silently; at the row level extra keys are lethal**, because
`event` / `placement` / `pallet_state` / `snapshot` take `**kwargs` and every key is a
column. Columns are SI; event payloads are in mm and degrees, and that conversion happens
at the telemetry boundary and nowhere else. `task` must be `"palletizing"`.

## Dependencies

Everything the simulation needs is pinned to `==` in
[`requirements.txt`](requirements.txt); the demonstrator declares its own, looser set in
[`tools/pyproject.toml`](tools/pyproject.toml) with a `uv.lock` beside it. Two rules: do
not add a dependency without first checking whether MuJoCo or NumPy already do the job,
and if you must, pin it and check its licence against MIT first.

| Package | Version | Licence | Used for |
|---|---|---|---|
| `mujoco` | 3.13.0 | Apache-2.0 | Physics, rendering, the viewer |
| `numpy` | 2.4.4 | BSD-3-Clause | Everywhere. The only thing `placing/` imports |
| `PyYAML` | 6.0.3 | MIT | `configs/` |
| `imageio` | 2.37.3 | BSD-2-Clause | Writing the pallet PNGs and the timelapse |
| `theker_telemetry` | editable | see Platform | The platform contract. **Not** in `requirements.txt`, on purpose |

`tools/` adds `pytest` and `ruff` (`--extra dev`) and `imageio-ffmpeg` (`--extra video`,
imported lazily and only when a video is written).

Third-party model attribution — the UR10e kinematics, inertias and meshes from MuJoCo
Menagerie, under BSD-3-Clause — is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The meshes are downloaded at install
time and never committed.

> **Status: `requirements.txt` has drifted.** `mink==1.3.0`, `qpsolvers==4.13.0`,
> `daqp==0.9.1` and `scipy==1.18.1` are still pinned, but nothing in `src/`, `placing/`,
> `scripts/` or `tests/` imports any of them — the IK is now the damped least-squares
> solver in `src/cell/arm.py`, ported from the demonstrator. They are installed and
> unused. Cleaning that up is separate work, and worth doing: `qpsolvers` is LGPL-3.0 and
> the only entry in the set that is not permissive.

## Licence

[MIT](LICENSE) © 2026 STACKSPECT.

The dependency audit above found no licence incompatible with releasing this project under
MIT.
