# Contributing

Thanks for looking. This repository is a skeleton under active construction: the
contracts are fixed and the architecture is written down, but almost every module is
still a docstring. Read [README.md](README.md) for what exists today, and read
**[AGENTS.md](AGENTS.md) in full before writing any code** — it is in Spanish, it is the
only place the project's real knowledge lives, and every hard rule in it cost a lost run
in the predecessor repository.

## Before anything else

**Read `AGENTS.md`.** Not as a courtesy. It carries the three closed vocabularies of the
platform contract and the traps that fail late and silently. If something there
contradicts what you believe, `AGENTS.md` wins; if something there is out of date, fix
it in the same PR.

## Branches

| Branch | What it is |
|---|---|
| `dev` | **The integration branch.** Everything lands here first. Branch from it, target it. |
| `main` | Release-ish. Synced from `dev`; do not develop against it. |

Name your branch `<author>/<topic>`, as the history already does
(`manuamest/docs-agents-md-arm`). Open the pull request against `dev`.

## Commits

Conventional commits, as used throughout the history of this repo and of
`Guionized-simulation`:

```
feat: paletizado guionizado, extraído a su propio repositorio
feat(telemetry): sube la cenital y el alzado a Supabase Storage
fix(cli): --telemetry sin Supabase falla en voz alta
refactor(telemetry): al contrato de dev, sin apaños
docs: arquitectura y esqueleto del paletizado real
chore: la ejecución se llama «paletizado guionizado» en la interfaz
```

Types in use: `feat`, `fix`, `refactor`, `docs`, `chore`, with an optional scope. Subject
lines are written in Spanish, in the imperative, lowercase, no trailing period.

**Language convention, so nobody "fixes" it in the wrong direction:** code, code
comments, docstrings, config files and commit subjects are **Spanish**; repository-level
documentation aimed at the outside world (`README.md`, `CONTRIBUTING.md`, `LICENSE`) is
**English**. Do not translate the Spanish comments — they carry measured values and the
reasoning behind them.

## Getting the environment up

```bash
# The Platform repo must be checked out as a sibling directory, on its `dev` branch.
git clone -b dev https://github.com/STACKSPECT/Platform ../Platform

bash scripts/setup.sh        # venv + deps + SDK + arm model
source .venv/bin/activate
```

`scripts/setup.sh` is the only executable path in the repo today, and it does four
things: creates `.venv`, installs `requirements.txt`, installs the `theker_telemetry`
SDK **editable from the Platform repo**, and clones `mujoco_menagerie` into
`third_party/`.

Point it elsewhere with the `PLATFORM` variable if your clone is not at `../Platform`:

```bash
PLATFORM=/path/to/Platform bash scripts/setup.sh
```

**It has to be Platform's `dev` branch.** `main` lacks `pallet.py` — where
`stability_margin` and `support_polygon` come from — and lacks the episode lifecycle.
The failure does not show up at install time, it shows up at run time, which is why
`setup.sh` asserts on it and stops.

For credentials, copy `.env.example` to `.env` (in this repo or its parent directory).
Without them the SDK does not complain: it treats "no credentials" as its normal mode
and writes to disk only.

## Dependencies

Everything is pinned to `==` in `requirements.txt`, reconciled against the set
`Guionized-simulation` already proved works together. Two rules:

- **Do not add a dependency** without first checking whether MuJoCo or NumPy already do
  it.
- If you must, pin it, and check its licence is compatible with MIT before you commit
  (see the README's dependency table).

`theker_telemetry` deliberately does not appear in `requirements.txt`: it is the contract
of whoever stores the data, so it is installed editable from Platform by `setup.sh`.

## How changes are verified

There is no CI. The verification ladder is defined in **`AGENTS.md` §9** and it is
ordered cheapest-first on purpose — run it in this order and stop at the first failure:

1. **No network, no simulator:** `python tests/test_pallet.py` — fails in seconds.
2. **The measurement alone:** `python -m src.measure` — its own asserts.
3. **Without Supabase:** `python scripts/palletize.py -n 1 --no-telemetry` — one whole
   episode to disk; inspect the `episodes.jsonl` under `runs/`.
4. **With telemetry, and it must say which mode it is in.** If it does not print that
   telemetry is active, it is not.
5. **With the platform UI open in a browser.** This is the real test: the episode shows
   up *in progress* within seconds, the pallet builds package by package, the KPIs move
   on their own.
6. **Against the database**, afterwards — the SQL checks are in `AGENTS.md` §9.

Note that **none of these six rungs can be run today** — `tests/test_pallet.py`,
`src/measure.py` and `scripts/palletize.py` are all still to be written. Until they
exist, the only thing a contributor can actually execute is `bash scripts/setup.sh`.

### The tests use no framework, on purpose

`tests/test_pallet.py` is meant to be **plain asserts run as a script**:

```bash
python tests/test_pallet.py
```

Not pytest. Do not add a framework, a `conftest.py`, or fixtures. The point is that the
checks run in seconds with nothing installed beyond the dependencies, and that anyone can
read the file top to bottom and see what is being anchored. The same applies to
`src/measure.py`, which is to ship its own `demo()` of asserts behind
`python -m src.measure`.

## The easiest thing to break

**The three closed vocabularies of the platform contract.** Inventing a value does not
raise an error where you write it — it fails later, silently, and usually takes the rest
of the run's uploads with it:

| Vocabulary | Allowed values | What happens if you invent one |
|---|---|---|
| `events.kind` | `perceive`, `plan`, `pick`, `place`, `settle`, `fail` | A closed CHECK of six values. The database rejects the row with a 400 the SDK swallows, and **uploads stay off for the rest of the run**. There is deliberately **no kind for the conveyor** — its state rides in the `perceive` payload. |
| `snapshots.view` | `top`, `side`, `iso`, `camera` | The PNG uploads to Storage and *then* the database rejects the row with a `23514`: an orphaned photo and a trace with no image. The previous repo hit this with `front`. |
| `failure` | `no_detection`, `ik_unreachable`, `collision`, `grasp_slip`, `wrong_placement`, `timeout`, `stack_collapse`, `overhang_violation` | `EpisodeResult` raises `ValueError` on anything else, on purpose. Adding one means touching three places in Platform — ask whoever owns the backend, do not invent it here. |

Related asymmetry, worth memorising: **inside an event `payload`, extra keys are
harmless and missing keys break silently. At the row level, extra keys are lethal** —
`event`/`placement`/`pallet_state`/`snapshot` take `**kwargs` and every key is a column,
so one typo is a 400 that turns uploads off. `task` is the level's source — `table`,
`conveyor` or `truck` — and comes from `scene.level.source`, never hard-coded.

`tests/test_pallet.py` exists to catch exactly these before you spend minutes of physics
finding out.

## Architectural boundaries

Four rules that keep the layers separable (the full list is `AGENTS.md` §8):

- **Platform knowledge lives only in `src/telemetry.py`.** No other module should know
  the name of any column. What crosses module boundaries are `contracts.py` objects, not
  rows.
- **Do not reimplement `stability_margin` or `support_polygon`.** They come from the SDK,
  and a third copy would drift from the one the UI uses.
- **Do not let a module read the scene on its own.** If `heuristic.py` imports `mujoco`,
  something went wrong — it needs a `Heightmap` and a `PackageSpec`. The oracle stubs are
  the deliberate exception, which is why they live in separate, clearly named files
  (`vision/oracle.py`, `planner/naive.py`).
- **Do not eyeball `configs/`.** Every odd value has its measurement written beside it.
  If you change one, write down how you measured it.

## Opening a pull request

- Target `dev`.
- Say which rungs of the verification ladder you ran, and what they printed.
- If you changed a calibrated value in `configs/`, include the measurement.
- If you learned something durable about the project, update `AGENTS.md` in the same PR.
