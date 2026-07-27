#!/usr/bin/env python3
"""Conservative screening, paired confirmation, and inclusive UCC ranges."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

from ucc_tune_runner import (RunResult, RunSpec, is_sizeless_collective,
                             measure, measure_paired)
from ucc_tune_space import (
    AlgInfo, CollectiveCountPlan, Knob, collective_count_plan,
    competition_env, dtype_size, is_team_independent, is_unkeyable_collective,
    knobs_for, score_map_key_quantum, tune_env_var,
)
from ucc_tune_stats import (CellKey, Decision, PairedEvidence, ProofBudget,
                            classify_cell)

logger = logging.getLogger(__name__)

# UCC TUNE grammar accepts the stringified enum suffix (underscore), not the
# hyphenated display name.  ucc_mem_type_from_str() (src/utils/ucc_coll_utils.c)
# display name in ucc_mem_type_str() (src/components/mc/base/ucc_mc_base.c) and
# is REJECTED by the score parser, which discards the entire TUNE value.
_PERFTEST_TO_TUNE_MEM = {
    "host": "host", "cuda": "cuda", "cuda-mng": "cuda_managed", "rocm": "rocm",
}


def _mem_type_for_tune(perftest_mem_type: str) -> str:
    try:
        return _PERFTEST_TO_TUNE_MEM[perftest_mem_type]
    except KeyError as exc:
        raise ValueError(f"Unknown perftest mem_type {perftest_mem_type!r}") from exc


def _fmt_bytes(n: int) -> str:
    for suffix, factor in (("G", 1 << 30), ("M", 1 << 20), ("k", 1 << 10)):
        if n >= factor and n % factor == 0:
            return f"{n // factor}{suffix}"
    return str(n)


@dataclasses.dataclass
class SweepSpec:
    component: str
    collective: str
    mem_type: str
    team_size: int
    msg_sizes_bytes: list
    alg_list: list
    all_team_sizes: list = ()
    datatype: str = "float32"
    reduction_op: str = "sum"
    n_reps: int = 7
    n_iter: int = 1000
    n_warmup: int = 100
    persistent: bool = True
    margin_threshold: float = 0.05
    min_pairs: int = 10
    max_pairs: int = 20
    boundary_resolution_bytes: int = 1024
    max_boundary_probes: int = 4
    max_confirmation_points: int = 40
    confirmation_seed: int = 0
    mpi_launcher: list = dataclasses.field(default_factory=lambda: ["mpirun", "-np", "1"])
    executed_team_size: Optional[int] = None
    perftest_path: str = "ucc_perftest"
    proof_mode: bool = False
    skip_knobs: bool = False
    timeout_s: int = 120
    readback_log_level: Optional[str] = "info"

    def __post_init__(self) -> None:
        if self.executed_team_size is None:
            self.executed_team_size = self.team_size
        if self.executed_team_size != self.team_size:
            raise ValueError(
                f"team-size mismatch: requested {self.team_size}, launcher "
                f"bound to {self.executed_team_size}")
        if self.min_pairs < 10 or self.max_pairs < self.min_pairs or self.max_pairs > 20:
            raise ValueError("pair limits must satisfy 10 <= min_pairs <= max_pairs <= 20")
        if self.boundary_resolution_bytes <= 0:
            raise ValueError("boundary resolution must be positive")
        if not 0 <= self.max_boundary_probes <= 12:
            raise ValueError("max boundary probes must be between 0 and 12")
        if self.max_confirmation_points < 0:
            raise ValueError("max confirmation points must be non-negative")


@dataclasses.dataclass
class SizeDecision:
    size_bytes: int
    should_override: bool
    winner_name: str
    winner_id: int
    winner_median_us: float
    default_median_us: Optional[float]
    margin: float
    knob_overrides: dict
    policy: Decision = Decision.DEFAULT
    evidence: Optional[PairedEvidence] = None
    actual_size_bytes: Optional[int] = None
    source: str = "screening"
    knob_hypotheses: list = dataclasses.field(default_factory=list)

    # Score-map readback (ROADMAP A1): what UCC's default arm actually
    # selected, and the measurement CV of the winning/default arms (A3).
    default_selected_alg: Optional[str] = None
    default_selected_component: Optional[str] = None
    winner_cv: Optional[float] = None
    default_cv: Optional[float] = None

    def __post_init__(self) -> None:
        if self.actual_size_bytes is None:
            self.actual_size_bytes = self.size_bytes
        if self.should_override and self.policy == Decision.DEFAULT:
            # Compatibility for explicitly constructed decisions; sweep_cell
            # itself sets this only from paired WIN evidence.
            self.policy = Decision.WIN


@dataclasses.dataclass
class TuneRange:
    """Finite inclusive byte range; open tails are intentionally unsupported."""
    start_bytes: int
    end_bytes: int
    alg_name: str
    alg_id: int
    knob_overrides: dict
    evidence_points: tuple = ()
    resolution_bytes: int = 1024

    def __post_init__(self) -> None:
        if self.end_bytes is None:
            raise ValueError("routine ranges require a finite inclusive end")
        if self.start_bytes < 0 or self.end_bytes < self.start_bytes:
            raise ValueError("invalid inclusive range")

    def contains(self, size_bytes: int) -> bool:
        return self.start_bytes <= size_bytes <= self.end_bytes

    @property
    def is_emittable(self) -> bool:
        """UCC's score-map grammar requires start < end
        (coll_score_add_range rejects start >= end). A zero-width range cannot
        be expressed; it is reported as a finding lead, never emitted."""
        return self.start_bytes < self.end_bytes

    def tune_token(self, collective: str, mem_type_tune: str,
                   team_low: int, team_high: Optional[int] = None) -> str:
        # An unqualified team observation is a singleton, never an open tail.
        high = team_low if team_high is None else team_high
        return (f"{collective}:{_fmt_bytes(self.start_bytes)}-"
                f"{_fmt_bytes(self.end_bytes)}:{mem_type_tune}:"
                f"[{team_low}-{high}]:inf:@{self.alg_name}")


def _compute_team_bands(team_sizes: list) -> dict[int, tuple[int, int]]:
    """Return exact measured team-size scopes; no interpolation or tail."""
    return {size: (size, size) for size in sorted(set(team_sizes))}


@dataclasses.dataclass
class SweepResult:
    spec: SweepSpec
    size_decisions: list
    tune_ranges: list
    warnings: list
    proof_budget: Optional[ProofBudget] = None
    unsupported_regimes: tuple[str, ...] = ()


@dataclasses.dataclass
class KnobHypothesis:
    """Auditable member of the cell-wide Holm family for knob attribution."""
    hypothesis_id: str
    env_var: str
    candidate: str
    gate: str
    evidence: PairedEvidence

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "env_var": self.env_var,
            "candidate": self.candidate,
            "gate": self.gate,
            "raw_p_value": self.evidence.p_win,
            "adjusted_threshold": self.evidence.adjusted_alpha,
            "decision": self.evidence.decision.value,
            "reason": self.evidence.reason,
            "evidence": self.evidence.to_dict(),
        }


def _forced_alg_env(spec: SweepSpec, alg_name: str) -> dict:
    env = competition_env(spec.component)
    env[tune_env_var(spec.component)] = (
        f"{spec.collective}:0-inf:{_mem_type_for_tune(spec.mem_type)}:"
        f"[1-inf]:inf:@{alg_name}"
    )
    return env


def _default_env(spec: SweepSpec) -> dict:
    return competition_env(spec.component)


def _cell_team(spec: SweepSpec) -> int:
    """Team size the cell's counts scale by.

    Collectives whose source/destination counts are team-scaled cannot be
    sized without knowing it: guessing one rank would mislabel every
    score-map key, so fail closed instead of measuring the wrong workload.
    """
    team = spec.executed_team_size
    if team is not None:
        return team
    if not is_team_independent(spec.collective):
        raise ValueError(
            f"{spec.collective} scales its counts by the team size; bind the "
            "cell to an executed team size before measuring it")
    return 1


def _plan(spec: SweepSpec, size_bytes: int) -> CollectiveCountPlan:
    """Count/score-map plan mapping a grid size onto one perftest run."""
    return collective_count_plan(spec.collective, spec.datatype, size_bytes,
                                  _cell_team(spec))


def _key_quantum(spec: SweepSpec) -> int:
    """Smallest distinguishable score-map step; boundary/probe alignment unit."""
    return score_map_key_quantum(spec.collective, spec.datatype,
                                 _cell_team(spec))


def _run_spec_for(spec: SweepSpec, size_bytes: int, extra_env: dict) -> RunSpec:
    return RunSpec(
        collective=spec.collective, mem_type=spec.mem_type,
        count=_plan(spec, size_bytes).count, datatype=spec.datatype,
        reduction_op=spec.reduction_op, n_reps=spec.n_reps,
        n_iter=spec.n_iter, n_warmup=spec.n_warmup,
        persistent=spec.persistent, extra_env=extra_env,
        mpi_launcher=list(spec.mpi_launcher), perftest_path=spec.perftest_path,
        requested_team_size=spec.team_size,
        executed_team_size=spec.executed_team_size,
        timeout_s=spec.timeout_s,
        readback_log_level=spec.readback_log_level,
        alg_names=[alg.name for alg in spec.alg_list],
    )


def _measure_safe(spec: SweepSpec, size_bytes: int, extra_env: dict,
                  label: str) -> Optional[RunResult]:
    try:
        return measure(_run_spec_for(spec, size_bytes, extra_env))
    except RuntimeError as exc:
        logger.warning("Measurement failed [%s size=%d]: %s", label, size_bytes, exc)
        return None


def _sweep_algs_at_size(spec: SweepSpec, size_bytes: int) -> dict:
    results = {}
    for alg in spec.alg_list:
        result = _measure_safe(spec, size_bytes, _forced_alg_env(spec, alg.name),
                               f"{spec.component}/{spec.collective}/@{alg.name}")
        if result is not None:
            results[alg.name] = result
    return results


def _paired_compare(spec: SweepSpec, size_bytes: int, default_env: dict,
                    candidate_env: dict, seed: int, *, default_arm="D",
                    candidate_arm="A") -> PairedEvidence:
    default_spec = _run_spec_for(spec, size_bytes, default_env)
    candidate_spec = _run_spec_for(spec, size_bytes, candidate_env)
    return measure_paired(
        default_spec, candidate_spec, seed=seed, min_pairs=spec.min_pairs,
        max_pairs=spec.max_pairs, min_speedup=spec.margin_threshold,
        default_arm=default_arm, candidate_arm=candidate_arm,
    ).evidence


def confirm_knob(spec: SweepSpec, size_bytes: int, alg_name: str,
                 knob: Knob, candidate: str, seed: int = 0
                 ) -> tuple[Optional[str], tuple[PairedEvidence, ...]]:
    """Apply the D/A0, A1/A0, and joint A1/D attribution gates."""
    d_env = _default_env(spec)
    a0_env = _forced_alg_env(spec, alg_name)
    a1_env = {**a0_env, knob.env_var: candidate}
    alg = _paired_compare(spec, size_bytes, d_env, a0_env, seed)
    knob_effect = _paired_compare(
        spec, size_bytes, a0_env, a1_env, seed + 1,
        default_arm="A0", candidate_arm="A1",
    )
    joint = _paired_compare(spec, size_bytes, d_env, a1_env, seed + 2)
    if all(e.decision == Decision.WIN for e in (alg, knob_effect, joint)):
        return candidate, (alg, knob_effect, joint)
    return None, (alg, knob_effect, joint)


def _decision_key(decision: SizeDecision) -> tuple:
    return (decision.winner_name, decision.winner_id,
            tuple(sorted(decision.knob_overrides.items())))


def coalesce_ranges(size_decisions: list, all_sizes: list,
                    resolution_bytes: int = 1024) -> list:
    """Build finite inclusive ranges from contiguous confirmed WIN points."""
    if not all_sizes:
        return []
    by_size = {d.actual_size_bytes: d for d in size_decisions}
    ordered_sizes = sorted(set(by_size))
    ranges: list[TuneRange] = []
    group: list[SizeDecision] = []

    def flush() -> None:
        nonlocal group
        if not group:
            return
        ranges.append(TuneRange(
            group[0].actual_size_bytes, group[-1].actual_size_bytes,
            group[0].winner_name, group[0].winner_id,
            dict(group[0].knob_overrides), tuple(group), resolution_bytes,
        ))
        group = []

    for size in ordered_sizes:
        decision = by_size[size]
        is_win = decision.should_override and decision.policy == Decision.WIN
        if not is_win:
            flush()
            continue
        if group and (_decision_key(group[-1]) != _decision_key(decision)
                      or size - group[-1].actual_size_bytes > resolution_bytes):
            flush()
        group.append(decision)
    flush()
    return ranges


def refine_boundaries(
    decisions: list[SizeDecision],
    confirm: Callable[[int, str, int], SizeDecision],
    *, datatype: str = "float32", resolution_bytes: int = 1024,
    align_bytes: Optional[int] = None, max_probes: int = 4,
    budget: Optional[ProofBudget] = None,
) -> tuple[list[SizeDecision], ProofBudget]:
    """Fixed-resolution refinement of unlike adjacent policy brackets.

    ``confirm(actual_bytes, candidate_name, candidate_id)`` must return fresh
    paired evidence.  Aligned duplicate byte counts are not re-measured.
    """
    if budget is None:
        budget = ProofBudget()
    align = dtype_size(datatype) if align_bytes is None else align_bytes
    by_size = {d.actual_size_bytes or d.size_bytes: d for d in decisions}
    original = sorted(by_size)
    transitions = [(original[i], original[i + 1]) for i in range(len(original) - 1)
                   if _decision_key(by_size[original[i]]) != _decision_key(by_size[original[i + 1]])
                   or by_size[original[i]].policy != by_size[original[i + 1]].policy]
    for left0, right0 in transitions:
        queue = [(left0, right0)]
        used = 0
        while queue and used < max_probes:
            left, right = queue.pop(0)
            if right - left <= resolution_bytes:
                continue
            midpoint = ((left + right) // 2 // align) * align
            if midpoint <= left:
                midpoint = left + align
            if midpoint >= right or midpoint in by_size:
                continue
            if budget.used_points >= budget.max_points:
                break
            side = by_size[left] if by_size[left].policy == Decision.WIN else by_size[right]
            candidate = side.winner_name
            candidate_id = side.winner_id
            decision = confirm(midpoint, candidate, candidate_id)
            decision.actual_size_bytes = midpoint
            decision.size_bytes = midpoint
            by_size[midpoint] = decision
            pairs = decision.evidence.complete_pairs if decision.evidence else 0
            budget = budget.consume(pairs)
            used += 1
            if (_decision_key(by_size[left]) != _decision_key(decision)
                    or by_size[left].policy != decision.policy):
                queue.append((left, midpoint))
            if (_decision_key(decision) != _decision_key(by_size[right])
                    or decision.policy != by_size[right].policy):
                queue.append((midpoint, right))
    return [by_size[size] for size in sorted(by_size)], budget


def _sweep_cell_screening(spec: SweepSpec) -> SweepResult:
    """Fast screening-only path: nominal winner vs default by median margin.

    No paired confirmation, boundary refinement, or knob attribution.  Emitted
    ranges are conservative singletons (or small groups) gated only by the
    screening margin.  This is the cheap discovery path; use --proof-mode for
    statistically confirmed emission.
    """
    if not spec.alg_list:
        return SweepResult(spec, [], [], [], None)
    warnings: list[str] = []
    decisions: list[SizeDecision] = []
    actual_seen: set[int] = set()

    for requested_size in spec.msg_sizes_bytes:
        actual_size = _plan(spec, requested_size).score_map_bytes
        if actual_size in actual_seen:
            warnings.append(f"aligned duplicate {requested_size} -> {actual_size} deduplicated")
            continue
        actual_seen.add(actual_size)
        alg_results = _sweep_algs_at_size(spec, actual_size)
        if not alg_results:
            warnings.append(f"All algorithms failed at {_fmt_bytes(actual_size)}")
            continue
        default_result = _measure_safe(spec, actual_size, _default_env(spec), "default screening")
        if len(alg_results) != len(spec.alg_list):
            warnings.append(f"partial algorithm sweep at {_fmt_bytes(actual_size)}")
        winner_name, winner_result = min(alg_results.items(), key=lambda item: item[1].median_us)
        winner = next(alg for alg in spec.alg_list if alg.name == winner_name)
        default_us = default_result.median_us if default_result else None
        margin = ((default_us - winner_result.median_us) / default_us
                  if default_us and default_us > 0 else 0.0)
        should_override = default_result is not None and margin > spec.margin_threshold
        policy = Decision.WIN if should_override else Decision.DEFAULT
        decisions.append(SizeDecision(
            actual_size, should_override, winner_name, winner.id,
            winner_result.median_us, default_us, margin, {}, policy, None,
            actual_size, "screening-margin",
            default_selected_alg=(
                default_result.selected_alg if default_result else None),
            default_selected_component=(
                default_result.selected_component if default_result else None),
            winner_cv=winner_result.cv,
            default_cv=default_result.cv if default_result else None,
        ))
        for name, result in alg_results.items():
            if result.variance_warning:
                warnings.append(f"High CV ({result.cv * 100:.1f}%) for {name} at {_fmt_bytes(actual_size)}")
            if result.selected_alg and result.selected_alg != name:
                warnings.append(
                    f"readback mismatch at {_fmt_bytes(actual_size)}: "
                    f"forced @{name} but UCC selected {result.selected_alg}")

    ranges = coalesce_ranges(decisions, spec.msg_sizes_bytes,
                             spec.boundary_resolution_bytes)
    unsupported = tuple(
        f"{spec.component}/{spec.collective} mem={spec.mem_type} "
        f"team={spec.team_size} size={decision.actual_size_bytes}: "
        f"{decision.policy.value} (screening-margin)"
        for decision in decisions if decision.policy != Decision.WIN
    )
    return SweepResult(spec, decisions, ranges, warnings, None, unsupported)


def sweep_cell(spec: SweepSpec) -> SweepResult:
    """Dispatch to the screening (default) or paired-confirmation (proof) path."""
    if is_sizeless_collective(spec.collective):
        # perftest reports count/size N/A (0) for this collective, so the
        # measured workload carries none of the grid size the decision, the
        # emitted range, and its validation probes would be keyed on.  Reject
        # rather than emit ranges that can never select the measured winner.
        raise ValueError(
            f"{spec.collective} has no message-size dimension (perftest reports "
            "N/A); byte-range tuning cannot express or validate it")
    if is_unkeyable_collective(spec.collective):
        # ucc_coll_args_msgsize() cannot derive this collective's size from the
        # local arguments (it returns UCC_MSG_SIZE_ASYMMETRIC), so every byte
        # range would collapse onto the single asymmetric sentinel entry and
        # could neither express nor validate the measured workload.
        raise ValueError(
            f"{spec.collective} has no locally derivable score-map size "
            "(ucc_coll_args_msgsize returns UCC_MSG_SIZE_ASYMMETRIC); "
            "byte-range tuning cannot express or validate it")
    _cell_team(spec)
    if spec.proof_mode:
        return _sweep_cell_proof(spec)
    return _sweep_cell_screening(spec)


def _sweep_cell_proof(spec: SweepSpec) -> SweepResult:
    if not spec.alg_list:
        return SweepResult(spec, [], [], [], ProofBudget(spec.max_confirmation_points,
                                                         spec.max_pairs))
    warnings: list[str] = []
    decisions: list[SizeDecision] = []
    budget = ProofBudget(spec.max_confirmation_points, spec.max_pairs)
    actual_seen: set[int] = set()

    for requested_size in spec.msg_sizes_bytes:
        actual_size = _plan(spec, requested_size).score_map_bytes
        if actual_size in actual_seen:
            warnings.append(f"aligned duplicate {requested_size} -> {actual_size} deduplicated")
            continue
        actual_seen.add(actual_size)
        alg_results = _sweep_algs_at_size(spec, actual_size)
        if not alg_results:
            warnings.append(f"All algorithms failed at {_fmt_bytes(actual_size)}")
            continue
        default_result = _measure_safe(spec, actual_size, _default_env(spec), "default screening")
        winner_name, winner_result = min(alg_results.items(), key=lambda item: item[1].median_us)
        winner = next(alg for alg in spec.alg_list if alg.name == winner_name)
        default_us = default_result.median_us if default_result else None
        margin = ((default_us - winner_result.median_us) / default_us
                  if default_us and default_us > 0 else 0.0)
        partial = len(alg_results) != len(spec.alg_list)
        evidence: Optional[PairedEvidence] = None
        policy = Decision.DEFAULT
        source = "screening-only"
        if partial:
            warnings.append(f"partial algorithm sweep at {_fmt_bytes(actual_size)}")
        elif default_result is None:
            warnings.append(f"missing default at {_fmt_bytes(actual_size)}")
        elif budget.used_points >= budget.max_points:
            warnings.append(f"confirmation budget exhausted at {_fmt_bytes(actual_size)}")
        else:
            evidence = _paired_compare(
                spec, actual_size, _default_env(spec),
                _forced_alg_env(spec, winner_name),
                spec.confirmation_seed + budget.used_points,
            )
            budget = budget.consume(evidence.complete_pairs)
            policy = evidence.decision
            source = "fresh-paired-confirmation"
        decisions.append(SizeDecision(
            actual_size, policy == Decision.WIN, winner_name, winner.id,
            winner_result.median_us, default_us, margin, {}, policy, evidence,
            actual_size, source,
            default_selected_alg=(
                default_result.selected_alg if default_result else None),
            default_selected_component=(
                default_result.selected_component if default_result else None),
            winner_cv=winner_result.cv,
            default_cv=default_result.cv if default_result else None,
        ))
        for name, result in alg_results.items():
            if result.variance_warning:
                warnings.append(f"High CV ({result.cv * 100:.1f}%) for {name} at {_fmt_bytes(actual_size)}")
            if result.selected_alg and result.selected_alg != name:
                warnings.append(
                    f"readback mismatch at {_fmt_bytes(actual_size)}: "
                    f"forced @{name} but UCC selected {result.selected_alg}")

    def confirm_boundary(size: int, candidate: str, candidate_id: int) -> SizeDecision:
        evidence = _paired_compare(
            spec, size, _default_env(spec), _forced_alg_env(spec, candidate),
            spec.confirmation_seed + 100 + size,
        )
        return SizeDecision(
            size, evidence.decision == Decision.WIN, candidate, candidate_id,
            float("nan"), None, 0.0, {}, evidence.decision, evidence, size,
            "boundary-paired-confirmation",
        )

    decisions, budget = refine_boundaries(
        decisions, confirm_boundary, datatype=spec.datatype,
        align_bytes=_key_quantum(spec),
        resolution_bytes=spec.boundary_resolution_bytes,
        max_probes=spec.max_boundary_probes, budget=budget,
    )

    # Freeze all evidence before making any emitted decision.  The complete
    # per-cell family consists of every collected algorithm anchor, refined
    # boundary, and D/A0, A1/A0, A1/D knob-attribution hypothesis.
    family: list[tuple[str, PairedEvidence, SizeDecision, Optional[KnobHypothesis],
                       tuple[str, str]]] = []
    for decision in decisions:
        if decision.evidence is not None:
            hypothesis_id = (
                f"algorithm:{decision.source}:{decision.actual_size_bytes}:"
                f"{decision.winner_name}"
            )
            family.append((hypothesis_id, decision.evidence, decision, None,
                           ("D", "A")))

    # Knob evidence is collected only at raw-WIN algorithm points, but no knob
    # or final algorithm decision is mutated until the one cell-wide Holm pass.
    # Skipped entirely when --no-knobs is set (algorithm-only confirmation).
    if not spec.skip_knobs:
        for decision in sorted(decisions, key=lambda item: item.actual_size_bytes):
            if decision.policy != Decision.WIN:
                continue
            knobs = sorted(knobs_for(spec.component, spec.collective, decision.winner_name),
                           key=lambda item: item.env_var)
            for knob_index, knob in enumerate(knobs):
                required_points = 3 * len(knob.candidates)
                if budget.used_points + required_points > budget.max_points:
                    warnings.append(f"confirmation budget prevents complete knob sweep for {knob.env_var}; omitted")
                    continue
                for candidate_index, candidate in enumerate(sorted(knob.candidates)):
                    _, evidence_set = confirm_knob(
                        spec, decision.actual_size_bytes, decision.winner_name,
                        knob, candidate,
                        spec.confirmation_seed + 1000 + knob_index * 100 + candidate_index * 3,
                    )
                    for gate, evidence in zip(("algorithm", "knob-effect", "joint"),
                                              evidence_set):
                        budget = budget.consume(evidence.complete_pairs)
                        hypothesis_id = (
                            f"knob:{decision.actual_size_bytes}:{knob.env_var}:"
                            f"{candidate}:{gate}"
                        )
                        audit = KnobHypothesis(hypothesis_id, knob.env_var,
                                               candidate, gate, evidence)
                        decision.knob_hypotheses.append(audit)
                        # The knob-effect gate measured its two conditions under
                        # the A0/A1 labels (confirm_knob renames them so the
                        # three gates stay distinguishable); the algorithm and
                        # joint gates are plain D/A comparisons.
                        family.append((
                            hypothesis_id, evidence, decision, audit,
                            ("A0", "A1") if gate == "knob-effect" else ("D", "A")))


    if family:
        adjusted = classify_cell(
            [item[1].samples for item in family],
            hypothesis_ids=[item[0] for item in family],
            arm_pairs=[item[4] for item in family],
            min_pairs=spec.min_pairs, min_speedup=spec.margin_threshold,
        )
        for (_, _, decision, audit, _arm_pair), evidence in zip(family, adjusted):
            if audit is None:
                decision.evidence = evidence
                decision.policy = evidence.decision
                decision.should_override = evidence.decision == Decision.WIN
            else:
                audit.evidence = evidence

    # Retain exactly one candidate only if its complete three-gate attribution
    # set survives Holm and the point's algorithm hypothesis also survives.
    for decision in decisions:
        if decision.policy != Decision.WIN:
            decision.knob_overrides.clear()
            continue
        by_knob: dict[str, dict[str, list[KnobHypothesis]]] = {}
        for audit in decision.knob_hypotheses:
            by_knob.setdefault(audit.env_var, {}).setdefault(audit.candidate, []).append(audit)
        for env_var, candidates in sorted(by_knob.items()):
            accepted = [candidate for candidate, gates in sorted(candidates.items())
                        if len(gates) == 3
                        and {gate.gate for gate in gates} == {"algorithm", "knob-effect", "joint"}
                        and all(gate.evidence.decision == Decision.WIN for gate in gates)]
            if len(accepted) == 1:
                decision.knob_overrides[env_var] = accepted[0]
            elif len(accepted) > 1:
                warnings.append(f"conflicting proven values for {env_var}; omitted")

    ranges = coalesce_ranges(decisions, spec.msg_sizes_bytes,
                             spec.boundary_resolution_bytes)
    unsupported = tuple(
        f"{spec.component}/{spec.collective} mem={spec.mem_type} "
        f"team={spec.team_size} size={decision.actual_size_bytes}: "
        f"{decision.policy.value} ({decision.evidence.reason if decision.evidence else decision.source})"
        for decision in decisions if decision.policy != Decision.WIN
    )
    return SweepResult(spec, decisions, ranges, warnings, budget, unsupported)
