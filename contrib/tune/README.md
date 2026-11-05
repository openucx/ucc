# UCC Offline Algorithm Tuner

Offline autotuner that measures UCC collective algorithm choices on *your*
platform with `ucc_perftest`, then emits a ready-to-ship tuning configuration
(`UCC_TUNE` score-map overrides) for the collectives, memory types, and team
sizes you care about.

## What it does

Given a built UCC (plus its `ucc_info` and `ucc_perftest`), the tuner runs
four stages:

1. **Fingerprint** — records UCC/UCX versions, CPU, GPU, and a config hash so
   results can be matched to the platform that produced them
   (`fingerprint.json`).
2. **Enumerate** — parses `ucc_info -A` to find every component/collective
   algorithm available for the requested collectives.
3. **Sweep** — for each cell (component x collective x mem_type x team_size),
   measures every algorithm (and, in `--proof-mode`, knob variants) at each
   message size with repeated `ucc_perftest` runs; rejects outliers with Tukey
   fences and keeps the median. An override is emitted only when the winner
   beats the UCC default by `--min-speedup` (default 5%).
4. **Validate** — replays representative sizes with the generated config
   versus the default and reports PASS/FAIL, so the output is checked to be
   *faster*, not merely *different*.

Decisions are coalesced into per-collective message-size ranges in UCC's
inclusive bracket syntax (e.g. `allreduce:4k-16k:@knomial+0:20`).

## Requirements

- Python **3.9+** (stdlib only — no third-party runtime dependencies).
- A built UCC with `ucc_info` and `ucc_perftest` on `PATH` (or pass
  `--ucc-info` / `--perftest` paths).
- An MPI launcher (`mpirun`/`srun`/`sbatch`-launched) able to start teams of
  `--team-sizes` ranks.
- For deploying long TUNE strings: UCC with `UCC_INI_MAX_LINE >= 8192`
  (commit 281a0eb5 or newer); older UCC silently truncates long config lines.

## Quick start

```sh
cd contrib/tune

# 1. Size the job first: prints planned invocations and estimated
#    wall-clock without launching anything.
python3 ucc_offline_tune.py --collective allreduce --team-sizes 8,64 \
    --min-bytes 8 --max-bytes 16777216 --dry-run

# 2. Run the sweep on an allocation.
python3 ucc_offline_tune.py --collective allreduce,allgather \
    --team-sizes 8,64,512 --mem-type host --output-dir ./out

# 3. Apply the result (provisional!) and benchmark your app.
export UCC_CONFIG_FILE=./out/ucc_tuned_provisional.conf
# or:  source ./out/ucc_tuned_provisional_env.sh
```

Run under SLURM with e.g. `salloc -N2 srun -n 64 python3
ucc_offline_tune.py --launcher "srun -n {team_size}" ...`. The launcher string
must contain exactly one rank-count option; its value is rebound per team size
(`--launcher "mpirun -np {team_size}"` is the default).

Key options (see `--help` for all):

| Option | Default | Meaning |
|---|---|---|
| `--collective` | `allreduce` | comma-separated collectives to tune |
| `--component` | auto | restrict to components, e.g. `tl/ucp,tl/cuda` |
| `--mem-type` | `host` | `host`, `cuda`, `cuda-mng` (comma-separated) |
| `--team-sizes` | `8` | comma-separated team sizes |
| `--min-bytes` / `--max-bytes` / `--factor` | 8 / 1 MiB / 2 | message-size grid |
| `--n-reps` / `--n-iter` / `--n-warmup` | 7 / 1000 / 100 | repetitions and perftest iterations |
| `--min-speedup` | 0.05 | required median speedup to emit an override |
| `--proof-mode` | off | full paired confirmation, boundary refinement, knob attribution (expensive) |
| `--readback-level` | `info` | parse the UCC score map to verify *which* component actually won (`debug` adds the algorithm) |
| `--no-validate` | off | skip Stage 4 validation (not recommended) |

## Output files (in `--output-dir`)

| File | Contents |
|---|---|
| `ucc_tuned_provisional.conf` | `UCC_CONFIG_FILE`-compatible overrides |
| `ucc_tuned_provisional_env.sh` | same settings as `export` lines |
| `fingerprint.json` | platform the measurements belong to |
| `results.json` | every cell's raw per-size decisions (for analysis) |
| `tuning_summary.txt` | human-readable per-cell log + skipped cells |
| `findings.md` / `findings.json` | ranked "leads": where the tuner saw wins, sized for triage |
| `knob_conflicts.txt` | knob conflicts across cells (only if any) |

The exit code is non-zero if any validation point failed or any cell produced
no measurements at all (a dead transport looks different from "default was
already best").

## Module map

```
ucc_offline_tune.py       driver, CLI, emission, validation (Stages 0-4)
ucc_tune_runner.py        ucc_perftest wrapper — the measurement oracle
ucc_tune_space.py         search space: ucc_info -A parsing, knobs, count plans
ucc_tune_sweep.py         per-cell coordinate-descent sweep + range coalescing
ucc_tune_stats.py         paired statistics (Tukey, medians, CI, decisions)
ucc_tune_fingerprint.py   platform fingerprint
test_ucc_*.py             unit tests (stubbed perftest; no cluster needed)
```

## Tests

```sh
cd contrib/tune && python3 -m pytest . -q      # or: python3 -m unittest discover .
```

All tests run against stubbed `ucc_info`/`ucc_perftest` output; no UCC
install or cluster is required.

## Limitations

- **Output is provisional by design.** The current driver has no input path
  for correctness evidence, so it never writes an "accepted" config.
  `ucc_perftest` does **not** check output buffers — validate collective
  correctness (MPI tests / gtest) under the tuned config before trusting it.
- Screening mode (the default) decides on median speedup alone; use
  `--proof-mode` for paired-confirmation statistics before deploying.
- `*v` / rooted / asymmetric collectives are skipped by default
  (their perftest size accounting makes range tuning unsound); sizeless
  collectives (no message-size dimension) are rejected outright.
- Results are platform-specific: re-tune after changing UCC/UCX version,
  topology, or team layout (the fingerprint exists to make staleness visible).
