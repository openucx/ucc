#!/usr/bin/env python3
"""
ucc_tune_runner.py — wraps ucc_perftest for the UCC offline tuner.

Corrections applied from plan addendum:
- Uses single-size runs (-b COUNT -e COUNT) because the exponential generator
  has a hard-coded ×2 factor that cannot be overridden at runtime via -f.
- Repeats N independent perftest invocations and computes median + IQR-based
  statistics; perftest itself only provides avg/min/max within a single run.
- Persistent mode (-p) is preferred to avoid per-iteration init/finalize and
  barrier overhead bleeding into the reported latency.
- CL/TL competition is controlled via extra_env: forcing UCC_TL_UCP_TUNE alone
  is not sufficient if NCCL/CUDA/HIER still score higher; callers must also set
  the appropriate disable/score env vars.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import random
import re
import statistics
import subprocess
from typing import Optional

from ucc_tune_stats import ArmSample, PairedEvidence, classify_evidence
from ucc_tune_space import (CollectiveCountPlan, count_plan_from_count,
                             is_team_independent)

logger = logging.getLogger(__name__)

_MPI_LAUNCHERS = frozenset({"mpirun", "mpiexec"})
_SRUN_LAUNCHERS = frozenset({"srun"})


def bind_launcher_team_size(launcher: list, team_size: int) -> tuple[list, int]:
    """Return a launcher bound to exactly ``team_size`` ranks, or fail closed."""
    if team_size <= 0 or not launcher:
        raise ValueError("team size and launcher must be non-empty and positive")
    kind = os.path.basename(launcher[0])
    if kind in _MPI_LAUNCHERS:
        separate, joined = {"-np", "-n"}, ("--np=", "--n=")
    elif kind in _SRUN_LAUNCHERS:
        separate, joined = {"-n", "--ntasks"}, ("--ntasks=",)
    else:
        raise ValueError(
            f"unsupported launcher {launcher[0]!r}; use mpirun/mpiexec or srun "
            "with one explicit rank-count option")
    matches: list[tuple[int, Optional[str]]] = []
    i = 1
    while i < len(launcher):
        token = launcher[i]
        if token in separate:
            if i + 1 >= len(launcher):
                raise ValueError(f"launcher rank option {token!r} has no value")
            matches.append((i + 1, None))
            i += 2
            continue
        prefix = next((p for p in joined if token.startswith(p)), None)
        if prefix is not None:
            matches.append((i, prefix))
        i += 1
    if len(matches) != 1:
        raise ValueError(
            "launcher must contain exactly one unambiguous rank-count option "
            "(-np/-n for mpirun, -n/--ntasks for srun)")
    index, prefix = matches[0]
    raw = launcher[index][len(prefix):] if prefix else launcher[index]
    if raw != "{team_size}":
        try:
            int(raw)
        except ValueError as exc:
            raise ValueError(
                f"launcher rank count {raw!r} is neither an integer nor "
                "{team_size}") from exc
    bound = list(launcher)
    bound[index] = f"{prefix or ''}{team_size}"
    return bound, team_size

# ---------------------------------------------------------------------------
# String maps matching perftest's CLI values exactly (ucc_pt_config.cc)
# ---------------------------------------------------------------------------

_VALID_COLLECTIVES = frozenset({
    "allgather", "allgatherv", "allreduce", "alltoall", "alltoallv",
    "barrier", "bcast", "gather", "gatherv", "reduce", "reduce_scatter",
    "reduce_scatterv", "scatter", "scatterv",
})

# Collectives that do not have a message-size dimension (print "N/A" for count/size).
_SIZELESS_COLLECTIVES = frozenset({"barrier"})

# Collectives that take a reduction op (-o flag).  perftest fills
# coll_args.op for each of these (tools/perf/ucc_pt_coll_*.cc set
# has_reduction_ and assign coll_args.op), and defaults to SUM when -o is
# omitted — so leaving one out silently benchmarks the wrong workload.
_REDUCTION_COLLECTIVES = frozenset(
    {"allreduce", "reduce", "reduce_scatter", "reduce_scatterv"})

_VALID_MEM_TYPES = frozenset({"host", "cuda", "rocm", "cuda-mng"})

_VALID_DATATYPES = frozenset({
    "int8", "uint8", "int16", "uint16", "float16", "bfloat16",
    "int32", "uint32", "float32", "float32_complex",
    "int64", "uint64", "float64", "float64_complex",
    "int128", "uint128", "float128", "float128_complex",
})

_VALID_OPS = frozenset({"sum", "prod", "min", "max", "avg"})


def is_sizeless_collective(collective: str) -> bool:
    """True when perftest reports N/A (count 0, size 0) for this collective.

    Such collectives have no message-size dimension: every requested size is
    measured as the same size-0 workload, so byte-keyed TUNE ranges can never
    select the measured winner at their recorded size.
    """
    return collective in _SIZELESS_COLLECTIVES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunSpec:
    """Complete specification for one measurement point."""

    collective: str
    mem_type: str = "host"

    # Element count (NOT bytes). For single-size runs, min and max are both
    # set to this value. Datatypes have different element sizes, so count is
    # the right unit here — perftest's -b/-e flags take element counts.
    count: int = 1024

    datatype: str = "float32"
    reduction_op: str = "sum"       # only sent for _REDUCTION_COLLECTIVES

    # Repetition and iteration counts.
    # n_reps: independent perftest invocations (we compute median across these).
    # n_iter: -n flag (perftest iterations per rep; sets both small and large).
    # n_warmup: -w flag.
    n_reps: int = 7
    n_iter: int = 1000
    n_warmup: int = 100

    persistent: bool = True         # -p; avoids init/finalize per iteration

    # Environment variables injected for each run (merged over os.environ).
    # Use this for TUNE strings and CL/TL competition control, e.g.:
    #   {"UCC_TL_UCP_TUNE": "allreduce:0-inf:cuda:inf:@sra_knomial",
    #    "UCC_CLS": "basic",
    #    "UCC_TLS": "ucp"}
    extra_env: dict = dataclasses.field(default_factory=dict)

    # MPI launcher prefix, e.g. ["mpirun", "-np", "8"] or
    # ["srun", "-n", "8", "--gpus-per-task=1"].
    mpi_launcher: list = dataclasses.field(
        default_factory=lambda: ["mpirun", "-np", "1"]
    )
    requested_team_size: Optional[int] = None
    executed_team_size: Optional[int] = None

    perftest_path: str = "ucc_perftest"
    timeout_s: int = 120            # per-rep wall-clock timeout

    # Fraction by which a rep's latency must exceed the median to be flagged
    # as an outlier (Tukey fence multiplier applied to IQR).
    tukey_k: float = 1.5

    # CV threshold above which a variance warning is raised.
    cv_warn_threshold: float = 0.10

    # Score-map readback (ROADMAP A1).  When non-None, _run_once injects
    # UCC_LOG_LEVEL=<level> (unless extra_env already sets it) and parses the
    # team-create score map to record which component/algorithm UCC selected.
    # "info" records the component; "debug" additionally records the algorithm
    # via the init-function symbol.  alg_names maps those symbols back to the
    # algorithm names from ucc_info -A.
    readback_log_level: Optional[str] = "info"
    alg_names: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SingleRunSample:
    """Raw timing from one perftest invocation (the per-run avg across ranks)."""
    count: int          # element count; 0 for sizeless collectives
    size_bytes: int     # byte size; 0 for sizeless collectives
    avg_us: float
    min_us: float
    max_us: float

    # Score-map readback (ROADMAP A1): the component/algorithm UCC actually
    # selected for this arm, parsed from the team-create score-map log.
    selected_component: Optional[str] = None
    selected_alg: Optional[str] = None


@dataclasses.dataclass
class RunResult:
    """Aggregated statistics across n_reps independent runs."""
    spec: RunSpec
    samples: list               # list[SingleRunSample], one per successful rep

    # Computed over avg_us values from clean (non-outlier) reps.
    median_us: float
    iqr_us: float               # Q3 - Q1
    cv: float                   # stddev / median; gauge of environmental noise

    clean_count: int            # reps retained after outlier rejection
    dropped_count: int          # reps rejected as outliers
    failed_count: int           # reps that failed to run or parse

    variance_warning: bool      # True if cv > spec.cv_warn_threshold

    # Mode across successful reps of the score-map readback (ROADMAP A1).
    selected_component: Optional[str] = None
    selected_alg: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PairedRunResult:
    """Bounded paired collection, including failed arms and its verdict."""
    samples: tuple[ArmSample, ...]
    complete_pairs: int
    attempts: int
    seed: int
    evidence: PairedEvidence


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

# Matches the data row from print_time().  Columns are setw(12) with fixed
# precision-2 float formatting.  count/size are integers or the literal "N/A".
_DATA_LINE_RE = re.compile(
    r"^\s*(?P<count>\d+|N/A)\s+"
    r"(?P<size>\d+|N/A)\s+"
    r"(?P<avg>[\d.]+)\s+"
    r"(?P<min>[\d.]+)\s+"
    r"(?P<max>[\d.]+)"
)


def _parse_output(stdout: str, collective: str) -> Optional[SingleRunSample]:
    """Extract the single timing row from a single-size perftest run."""
    for line in stdout.splitlines():
        m = _DATA_LINE_RE.match(line)
        if m is None:
            continue
        count_str = m.group("count")
        size_str = m.group("size")
        count = 0 if count_str == "N/A" else int(count_str)
        size = 0 if size_str == "N/A" else int(size_str)
        return SingleRunSample(
            count=count,
            size_bytes=size,
            avg_us=float(m.group("avg")),
            min_us=float(m.group("min")),
            max_us=float(m.group("max")),
        )
    logger.debug("No data line found in perftest output:\n%s", stdout[:1000])
    return None


# ---------------------------------------------------------------------------
# Score-map readback (ROADMAP A1)
# ---------------------------------------------------------------------------

# UCC prints the resolved score map once at team create (ucc_team.c:471) when
# UCC_LOG_LEVEL >= info.  Every line carries the standard UCC log prefix:
#
#   [ts] [host:pid:tid] ucc_coll_score_map.c:225  UCC  INFO  Allreduce:
#   [ts] [host:pid:tid] ucc_coll_score_map.c:225  UCC  INFO  \tHost: {0..4095}:TL_UCP:10 {4K..inf}:TL_UCP:10
#
# At debug level each entry also carries the selected init function's symbol
# (get_fn_name in ucc_coll_score_map.c, via dladdr):
#   {0..4095}:TL_UCP:10=ucc_tl_ucp_allreduce_knomial_init
#
# Collective names come from ucc_coll_type_str() (capitalised "Allreduce",
# "Reduce_scatter"); memory types from ucc_mem_type_str() ("Host", "Cuda",
# "CudaManaged", "Rocm"); the component is log_component.name ("TL_UCP",
# "CL_BASIC").  Ranges use ucs_memunits_range_str() with a ".." separator and
# uppercase memunits suffixes ("0..4095", "4K..inf", "0..inf").

# UCC log prefix: "[<ts>] [<host>:<pid>:<tid>] <file>:<line>  UCC  <LEVEL>  "
_LOG_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s+\[[^\]]*\]\s+\S+:\d+\s+UCC\s+\S+\s+")

_COLL_SCORE_MAP_START_RE = re.compile(r"^=+\s*COLL_SCORE_MAP\b")
_COLL_SCORE_MAP_END_RE = re.compile(r"^=+\s*$")
_SCORE_MAP_COLL_RE = re.compile(r"^([A-Za-z_]+):\s*$")
_SCORE_MAP_MEM_RE = re.compile(r"^\s*([A-Za-z_]+):\s*(.*)$")
_SCORE_MAP_ENTRY_RE = re.compile(r"\{([^}]+)\}:([^:]+):([^:=\s]+)(?:=(\S+))?")

_MEMUNITS_SUFFIX = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}

_MEM_TYPE_LOG_TO_PERFTEST = {
    "host": "host",
    "cuda": "cuda",
    "cudamanaged": "cuda-mng",
    "rocm": "rocm",
    "rocmmanaged": "rocm-mng",
}


def _strip_log_prefix(line: str) -> str:
    """Strip the UCC log prefix, leaving the bare message (idempotent)."""
    return _LOG_PREFIX_RE.sub("", line, count=1)


def _normalize_component(name: str) -> str:
    """Normalize a UCC log_component.name ("TL_UCP") to the tuner spelling ("tl/ucp")."""
    return name.strip().lower().replace("_", "/")


def _parse_memunits(token: str) -> float:
    """Parse a UCC/UCX memunits token ("inf", "4K", "16", "1M") to a byte count."""
    token = token.strip().lower()
    if token == "inf":
        return math.inf
    m = re.fullmatch(r"(\d+)([kmgt]?)", token)
    if m is None:
        raise ValueError(f"unparseable memunits token {token!r}")
    return int(m.group(1)) * _MEMUNITS_SUFFIX[m.group(2)]


def parse_score_map(
    output: str,
    collective: str,
    mem_type: str,
    size_bytes: int,
) -> Optional[tuple[str, Optional[str]]]:
    """Return (component, fn_name) for the score-map entry covering size_bytes.

    ``collective`` and ``mem_type`` use the perftest spelling ("allreduce",
    "cuda-mng"); ``size_bytes`` is the actual measured byte size.  Returns None
    when the score map is absent or no entry covers the size.
    """
    in_block = False
    coll: Optional[str] = None
    for raw_line in output.splitlines():
        line = _strip_log_prefix(raw_line)
        if not in_block:
            if _COLL_SCORE_MAP_START_RE.match(line):
                in_block = True
            continue
        if _COLL_SCORE_MAP_END_RE.match(line):
            break
        m = _SCORE_MAP_COLL_RE.match(line)
        if m:
            coll = m.group(1).lower()
            continue
        m = _SCORE_MAP_MEM_RE.match(line)
        if m is None:
            continue
        mem_token = m.group(1).strip().lower()
        mem = _MEM_TYPE_LOG_TO_PERFTEST.get(mem_token, mem_token)
        if coll != collective or mem != mem_type:
            continue
        for entry in m.group(2).split():
            em = _SCORE_MAP_ENTRY_RE.match(entry)
            if em is None:
                continue
            parts = em.group(1).split("..")
            if len(parts) == 1:
                start = end = _parse_memunits(parts[0])
            else:
                start, end = _parse_memunits(parts[0]), _parse_memunits(parts[1])
            if start <= size_bytes <= end:
                return _normalize_component(em.group(2)), em.group(4)
    return None


def _alg_name_from_fn(fn_name: Optional[str], alg_names: list) -> Optional[str]:
    """Map a UCC init-function symbol to a known algorithm name, or None.

    The score-map debug print appends the init function's symbol, e.g.
    "ucc_tl_ucp_allreduce_sra_knomial_init".  The algorithm name is the longest
    known name appearing as an underscore-delimited token sequence in the
    symbol, so "sra_knomial" is preferred over "knomial".  A "?" symbol (dladdr
    failure) or an unknown symbol maps to None.
    """
    if not fn_name or not alg_names:
        return None
    matches = [name for name in alg_names
               if (f"_{name}_" in fn_name
                   or fn_name.endswith(f"_{name}_init")
                   or fn_name.endswith(f"_{name}_start")
                   or fn_name == name)]
    return max(matches, key=len) if matches else None


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _build_cmd(spec: RunSpec) -> list:
    """Return the perftest argv (without the MPI launcher prefix)."""
    if spec.collective not in _VALID_COLLECTIVES:
        raise ValueError(f"Unknown collective: {spec.collective!r}")
    if spec.mem_type not in _VALID_MEM_TYPES:
        raise ValueError(f"Unknown mem_type: {spec.mem_type!r}. Valid: {sorted(_VALID_MEM_TYPES)}")
    if spec.datatype not in _VALID_DATATYPES:
        raise ValueError(f"Unknown datatype: {spec.datatype!r}")
    if spec.reduction_op not in _VALID_OPS:
        raise ValueError(f"Unknown reduction_op: {spec.reduction_op!r}")

    cmd = [
        spec.perftest_path,
        "-c", spec.collective,
        "-b", str(spec.count),
        "-e", str(spec.count),   # min == max → single-size run
        "-m", spec.mem_type,
        "-d", spec.datatype,
        "-n", str(spec.n_iter),
        "-w", str(spec.n_warmup),
    ]

    if spec.collective in _REDUCTION_COLLECTIVES:
        cmd += ["-o", spec.reduction_op]

    if spec.persistent:
        cmd += ["-p"]

    return cmd


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def _plan_for(spec: RunSpec) -> CollectiveCountPlan:
    """The measurement's count/score-map/print plan for an instantiated spec.

    A spec without a bound team size is only sound for collectives whose
    counts do not scale with the team; for the team-scaled ones, assuming a
    single rank would mislabel every score-map key, so fail closed instead.
    """
    team = spec.executed_team_size or spec.requested_team_size
    if team is None:
        if not is_team_independent(spec.collective):
            raise ValueError(
                f"{spec.collective} is team-scaled: its score-map key depends "
                "on the team size, so requested/executed team size must be set "
                "before measuring it")
        team = 1
    return count_plan_from_count(spec.collective, spec.datatype, spec.count,
                                  team)


def _run_once(spec: RunSpec) -> Optional[SingleRunSample]:
    """Run perftest once and return the parsed timing sample, or None on failure."""
    if (spec.requested_team_size is not None
            and spec.requested_team_size != spec.executed_team_size):
        raise RuntimeError(
            f"team-size mismatch before launch: requested "
            f"{spec.requested_team_size}, bound {spec.executed_team_size}")
    env = os.environ.copy()
    env.update(spec.extra_env)
    if spec.readback_log_level is not None:
        # extra_env wins if the caller pinned UCC_LOG_LEVEL explicitly.
        env.setdefault("UCC_LOG_LEVEL", spec.readback_log_level)

    cmd = list(spec.mpi_launcher) + _build_cmd(spec)
    logger.debug("cmd: %s", " ".join(cmd))
    logger.debug("extra_env: %s", spec.extra_env)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=spec.timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("perftest timed out after %ds", spec.timeout_s)
        return None
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not launch {cmd[0]!r}: {exc}. "
            "Check mpi_launcher and perftest_path."
        ) from exc

    if proc.returncode != 0:
        logger.warning(
            "perftest exited %d\nstdout: %s\nstderr: %s",
            proc.returncode,
            proc.stdout[:500],
            proc.stderr[:500],
        )
        return None

    sample = _parse_output(proc.stdout, spec.collective)
    if sample is None:
        logger.warning("Could not parse perftest output:\n%s", proc.stdout[:500])
        return sample
    if spec.readback_log_level is not None:
        # UCC matches the score map against ucc_coll_args_msgsize(), which for
        # several collectives is neither the printed Size column nor count*dt
        # (see CollectiveCountPlan).  Probe with the key it actually uses.
        hit = parse_score_map(
            proc.stderr + "\n" + proc.stdout,
            spec.collective, spec.mem_type, _plan_for(spec).score_map_bytes,
        )
        if hit is not None:
            component, fn_name = hit
            sample.selected_component = component
            if fn_name:
                sample.selected_alg = _alg_name_from_fn(fn_name, spec.alg_names)
    return sample


def _tukey_clean(values: list, k: float) -> tuple[list, int]:
    """
    Return (clean_values, n_dropped) using Tukey fences.
    Points outside [Q1 - k*IQR, Q3 + k*IQR] are dropped.
    Falls back to returning all values if len < 4.
    """
    if len(values) < 4:
        return list(values), 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    clean = [v for v in values if lo <= v <= hi]
    dropped = len(values) - len(clean)
    if dropped:
        logger.debug(
            "Outlier rejection: dropped %d/%d reps (IQR=%.2f, fence=[%.2f, %.2f])",
            dropped, len(values), iqr, lo, hi,
        )
    return clean, dropped


def _mode(values) -> Optional[str]:
    """Most common non-None value, or None when all values are None/empty."""
    counts: dict = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def measure(spec: RunSpec) -> RunResult:
    """
    Run perftest spec.n_reps times, collect avg_us from each run, apply
    outlier rejection, and return aggregated statistics.

    Raises RuntimeError if fewer than 2 reps succeed (not enough data for
    meaningful statistics).
    """
    samples: list[SingleRunSample] = []
    failed = 0

    for rep in range(spec.n_reps):
        sample = _run_once(spec)
        if sample is None:
            failed += 1
            logger.warning("Rep %d/%d failed", rep + 1, spec.n_reps)
        else:
            samples.append(sample)
            logger.debug(
                "Rep %d/%d: avg=%.2f min=%.2f max=%.2f us",
                rep + 1, spec.n_reps,
                sample.avg_us, sample.min_us, sample.max_us,
            )

    if len(samples) < 2:
        raise RuntimeError(
            f"Only {len(samples)} of {spec.n_reps} reps succeeded "
            f"({failed} failed). Cannot compute statistics."
        )

    avg_times = [s.avg_us for s in samples]
    clean_times, dropped = _tukey_clean(avg_times, spec.tukey_k)

    if len(clean_times) < 2:
        logger.warning(
            "Outlier rejection removed too many reps; using all %d samples",
            len(samples),
        )
        clean_times = avg_times
        dropped = 0

    med = statistics.median(clean_times)
    stdev = statistics.stdev(clean_times) if len(clean_times) > 1 else 0.0
    cv = stdev / med if med > 0 else 0.0

    q_vals = sorted(clean_times)
    n = len(q_vals)
    iqr = q_vals[(3 * n) // 4] - q_vals[n // 4] if n >= 4 else 0.0

    result = RunResult(
        spec=spec,
        samples=samples,
        median_us=med,
        iqr_us=iqr,
        cv=cv,
        clean_count=len(clean_times),
        dropped_count=dropped,
        failed_count=failed,
        variance_warning=cv > spec.cv_warn_threshold,
        selected_component=_mode(s.selected_component for s in samples),
        selected_alg=_mode(s.selected_alg for s in samples),
    )

    if result.variance_warning:
        logger.warning(
            "High variance for %s count=%d mem=%s: CV=%.1f%% (threshold %.1f%%)",
            spec.collective, spec.count, spec.mem_type,
            cv * 100, spec.cv_warn_threshold * 100,
        )

    return result


def _balanced_orders(seed: int, count: int) -> list[str]:
    """Return a reproducible order sequence whose strata differ by at most one."""
    rng = random.Random(seed)
    orders: list[str] = []
    while len(orders) < count:
        block = ["AB", "BA"]
        rng.shuffle(block)
        orders.extend(block)
    return orders[:count]


def measure_paired(
    default_spec: RunSpec,
    candidate_spec: RunSpec,
    *,
    seed: int = 0,
    min_pairs: int = 10,
    max_pairs: int = 20,
    first_batch_attempts: int = 14,
    max_attempts: int = 28,
    min_speedup: float = 0.05,
    cv_threshold: float = 0.10,
    log_iqr_threshold: float = 0.10,
    default_arm: str = "D",
    candidate_arm: str = "A",
) -> PairedRunResult:
    """Collect fresh balanced AB/BA pairs without deleting outliers.

    ``AB`` means default then candidate and ``BA`` means candidate then default.
    The first batch stops after ``min_pairs`` complete pairs or 14 attempts.  A
    noisy first batch is extended, without inspecting its effect verdict, to
    the bounded 20-complete-pair/28-attempt ceiling.
    """
    if min_pairs < 10:
        raise ValueError("min_pairs below the safe minimum of 10")
    if max_pairs < min_pairs or max_pairs > 20:
        raise ValueError("max_pairs must be between min_pairs and 20")
    if first_batch_attempts < min_pairs or first_batch_attempts > max_attempts:
        raise ValueError("invalid first-batch attempt limit")
    if max_attempts > 28:
        raise ValueError("max_attempts exceeds the safe ceiling of 28")

    orders = _balanced_orders(seed, max_attempts)
    samples: list[ArmSample] = []
    complete = 0
    attempts = 0
    extend_for_variance = False

    def collect_arm(pair_id: int, order: str, arm: str, spec: RunSpec) -> ArmSample:
        sample = _run_once(spec)
        if sample is None:
            return ArmSample(pair_id, order, arm, None, False,
                             "measurement failed", seed)
        # The Count column is generator->get_src_count(), which perftest
        # team-scales for several collectives; comparing it to spec.count
        # rejected every team-scaled sample at team > 1 and accepted samples
        # whose workload differed from the one intended.  Compare against the
        # collective's expected printed count, and attribute the sample to the
        # score-map key size rather than the printed size.
        plan = _plan_for(spec)
        if sample.count != plan.expected_print_count:
            return ArmSample(pair_id, order, arm, sample.avg_us, False,
                             "wrong element count", seed, plan.score_map_bytes)
        if not math.isfinite(sample.avg_us) or sample.avg_us <= 0:
            return ArmSample(pair_id, order, arm, sample.avg_us, False,
                             "non-positive timing", seed, plan.score_map_bytes)
        return ArmSample(pair_id, order, arm, sample.avg_us, True, None,
                         seed, plan.score_map_bytes)

    while attempts < max_attempts and complete < max_pairs:
        if attempts >= first_batch_attempts and complete < min_pairs:
            break
        if complete >= min_pairs and not extend_for_variance:
            break
        order = orders[attempts]
        pair_id = attempts
        arm_specs = ((default_arm, default_spec), (candidate_arm, candidate_spec))
        if order == "BA":
            arm_specs = tuple(reversed(arm_specs))
        pair_samples = [collect_arm(pair_id, order, arm, arm_spec)
                        for arm, arm_spec in arm_specs]
        samples.extend(pair_samples)
        attempts += 1
        if all(sample.ok for sample in pair_samples):
            complete += 1

        if complete == min_pairs:
            provisional = classify_evidence(
                samples, min_pairs=min_pairs, min_speedup=min_speedup,
                cv_threshold=cv_threshold, require_stable_variance=False,
                log_iqr_threshold=log_iqr_threshold,
                default_arm=default_arm, candidate_arm=candidate_arm,
            )
            extend_for_variance = bool(
                provisional.cv_default is not None
                and (provisional.cv_default > cv_threshold
                     or provisional.cv_candidate > cv_threshold
                     or provisional.log_ratio_iqr > log_iqr_threshold)
            )

    evidence = classify_evidence(
        samples, min_pairs=min_pairs, min_speedup=min_speedup,
        cv_threshold=cv_threshold, require_stable_variance=True,
        log_iqr_threshold=log_iqr_threshold,
        default_arm=default_arm, candidate_arm=candidate_arm,
    )
    return PairedRunResult(tuple(samples), complete, attempts, seed, evidence)


# ---------------------------------------------------------------------------
# CLI entry point — standalone single-point test
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Measure a single (collective, count, mem_type) point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--collective", default="allreduce",
                        choices=sorted(_VALID_COLLECTIVES))
    parser.add_argument("-b", "--count", type=int, default=1024,
                        help="Element count")
    parser.add_argument("-m", "--mem-type", default="host",
                        choices=sorted(_VALID_MEM_TYPES))
    parser.add_argument("-d", "--datatype", default="float32",
                        choices=sorted(_VALID_DATATYPES))
    parser.add_argument("-o", "--op", default="sum",
                        choices=sorted(_VALID_OPS))
    parser.add_argument("-n", "--n-iter", type=int, default=1000)
    parser.add_argument("-w", "--n-warmup", type=int, default=100)
    parser.add_argument("-R", "--n-reps", type=int, default=7,
                        help="Independent perftest repetitions")
    parser.add_argument("--no-persistent", action="store_true",
                        help="Disable persistent mode (includes init/finalize overhead)")
    parser.add_argument("--perftest", default="ucc_perftest",
                        help="Path to ucc_perftest binary")
    parser.add_argument("--launcher", default="mpirun -np 1",
                        help="MPI launcher prefix as a space-separated string")
    parser.add_argument("--env", action="append", default=[],
                        metavar="KEY=VAL",
                        help="Extra env vars (may repeat); e.g. UCC_TLS=ucp")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    extra_env: dict = {}
    for kv in args.env:
        if "=" not in kv:
            parser.error(f"--env must be KEY=VAL, got: {kv!r}")
        k, v = kv.split("=", 1)
        extra_env[k] = v

    spec = RunSpec(
        collective=args.collective,
        mem_type=args.mem_type,
        count=args.count,
        datatype=args.datatype,
        reduction_op=args.op,
        n_reps=args.n_reps,
        n_iter=args.n_iter,
        n_warmup=args.n_warmup,
        persistent=not args.no_persistent,
        extra_env=extra_env,
        mpi_launcher=args.launcher.split(),
        perftest_path=args.perftest,
    )

    try:
        result = measure(spec)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print a concise summary to stdout.
    print(f"collective  : {spec.collective}")
    print(f"mem_type    : {spec.mem_type}")
    print(f"count       : {spec.count}")
    if result.samples:
        print(f"size_bytes  : {result.samples[0].size_bytes}")
    print(f"reps        : {result.clean_count}/{spec.n_reps} clean "
          f"({result.dropped_count} outliers, {result.failed_count} failed)")
    print(f"median_us   : {result.median_us:.2f}")
    print(f"iqr_us      : {result.iqr_us:.2f}")
    print(f"cv          : {result.cv * 100:.1f}%"
          + ("  [WARNING: high variance]" if result.variance_warning else ""))

    if args.verbose:
        raw = [{"avg_us": s.avg_us, "min_us": s.min_us, "max_us": s.max_us}
               for s in result.samples]
        print("raw_samples :", json.dumps(raw, indent=2))


if __name__ == "__main__":
    _cli_main()
