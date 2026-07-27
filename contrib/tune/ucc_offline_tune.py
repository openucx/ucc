#!/usr/bin/env python3
"""
ucc_offline_tune.py — top-level driver for the UCC offline algorithm tuner.

Implements the full Stages 0–4 of the offline tuner plan:

  Stage 0  Fingerprint the platform.
  Stage 1  Enumerate the search space via ucc_info -A.
  Stage 2  Sweep algorithms + knobs for each cell (via sweep_cell).
  Stage 3  Coalesce decisions into TUNE ranges (done inside sweep_cell).
  Stage 4  Validate: run the generated config vs. UCC default on a few
           representative sizes; confirm it is faster (not just different).

Output files written to --output-dir:
  ucc_tuned.conf        UCC_CONFIG_FILE-compatible config (KEY=VALUE).
  ucc_tuned_env.sh      Shell script: source to export the same vars.
  fingerprint.json      Platform fingerprint used to tag this run.
  results.json          All SweepResult data (for later analysis).
  tuning_summary.txt    Human-readable per-cell log and skipped-items list.

Correctness note (from plan addendum):
  ucc_perftest does not validate output buffers — the -c flag selects the
  collective, it is NOT a correctness check.  Run MPI or gtest coverage
  separately before deploying generated configs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shlex
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ucc_tune_fingerprint import Fingerprint, collect as collect_fingerprint
from ucc_tune_runner import (RunSpec, bind_launcher_team_size,
                             is_sizeless_collective, measure, measure_paired)
from ucc_tune_space import (
    bytes_to_count,
    collective_count_plan,
    dtype_size,
    is_unkeyable_collective,
    knob_metadata,
    msg_size_grid,
    parse_ucc_info_algs,
    run_ucc_info_algs,
    run_ucc_info_raw,
    score_map_key_quantum,
    tune_env_var,
)
from ucc_tune_stats import Decision, PairedEvidence
from ucc_tune_sweep import (
    SweepResult,
    SweepSpec,
    TuneRange,
    _compute_team_bands,
    _fmt_bytes,
    _mem_type_for_tune,
    sweep_cell,
)

logger = logging.getLogger(__name__)

# Collectives to skip by default: *v variants and rooted collectives.
# Rationale: size-0 lookup collapse makes message-range tuning dubious for these.
_ASYMMETRIC_COLLS = frozenset({
    "allgatherv", "alltoallv", "reduce_scatterv",
    "gatherv", "scatterv",
    "reduce", "gather", "scatter", "bcast",
})


@dataclasses.dataclass(frozen=True, order=True)
class CellKey:
    """Complete identity of one independently swept and validated cell."""

    component: str
    collective: str
    mem_type: str
    team_size: int
    datatype: str
    reduction_op: str


def _spec_cell_key(spec: SweepSpec) -> CellKey:
    return CellKey(spec.component, spec.collective, spec.mem_type,
                   spec.team_size, spec.datatype, spec.reduction_op)


def _point_cell_key(point: "ValidationPoint") -> Optional[CellKey]:
    """Return a key only for evidence executed at its requested team size."""
    requested = point.requested_team_size
    if (requested is None or requested != point.executed_team_size
            or point.component is None):
        return None
    return CellKey(point.component, point.collective, point.mem_type, requested,
                   point.datatype, point.reduction_op)


def _assert_emission_identity_compatible(results: list) -> None:
    """Reject mixed cells whose datatype/op cannot be encoded in UCC TUNE."""
    scopes: dict[tuple[str, str, str, int], CellKey] = {}
    for result in results:
        key = _spec_cell_key(result.spec)
        scope = (key.component, key.collective, key.mem_type, key.team_size)
        previous = scopes.setdefault(scope, key)
        if previous != key:
            raise ValueError(
                "cannot emit datatype/reduction-op variants into the same "
                f"UCC TUNE scope: {previous} and {key}")


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------

def _collect_tune_tokens(
    results: list,     # list[SweepResult]
) -> dict:            # tune_var → list[str token]
    """
    Aggregate all TuneRange tokens across cells, grouped by component TUNE var.
    Uses non-overlapping team-size bands when multiple team sizes are present.
    """
    _assert_emission_identity_compatible(results)
    tokens: dict[str, list[str]] = {}
    for result in results:
        if not result.tune_ranges:
            continue
        spec = result.spec
        tune_var = tune_env_var(spec.component)
        mt = _mem_type_for_tune(spec.mem_type)
        bands = _compute_team_bands(spec.all_team_sizes or [spec.team_size])
        team_low, team_high = bands.get(spec.team_size, (spec.team_size, None))
        for tr in result.tune_ranges:
            if not tr.is_emittable:
                logger.warning(
                    "Skipping zero-width range %s:%s-%s (UCC score-map grammar "
                    "rejects start>=end); reported as a finding lead only",
                    spec.collective, _fmt_bytes(tr.start_bytes),
                    _fmt_bytes(tr.end_bytes),
                )
                continue
            tok = tr.tune_token(spec.collective, mt, team_low, team_high)
            tokens.setdefault(tune_var, []).append(tok)
    return tokens


def _collect_knob_overrides(
    results: list,     # list[SweepResult]
) -> tuple[dict, list[str]]:
    """Emit range-safe knobs and omit every conflict or unscopable scalar."""
    _assert_emission_identity_compatible(results)
    artifact_cells = {_spec_cell_key(result.spec) for result in results}
    # env_var -> (start, end, mem, value, team, requested_min, requested_max)
    seen: dict[str, list[tuple[int, int, str, str, int, int, int]]] = {}
    for result in results:
        ts = result.spec.team_size
        domain = sorted(set(result.spec.msg_sizes_bytes))
        if not domain:
            continue
        for tr in result.tune_ranges:
            if not tr.is_emittable:
                continue
            for env_var, val in tr.knob_overrides.items():
                seen.setdefault(env_var, []).append((
                    tr.start_bytes, tr.end_bytes,
                    _mem_type_for_tune(result.spec.mem_type), val, ts,
                    domain[0], domain[-1],
                ))

    knob_env: dict[str, str] = {}
    warnings: list[str] = []
    for env_var, entries in sorted(seen.items()):
        metadata = knob_metadata(env_var)
        teams = {entry[4] for entry in entries}
        if len(teams) != 1:
            warning = f"Knob {env_var} cannot be team-scoped; omitted"
            warnings.append(warning)
            continue
        if metadata is not None and metadata.range_scoped:
            # Message/memory entries are explicit. Overlap with different
            # values is a conflict, not an invitation to pick a larger span.
            conflict = False
            for i, left in enumerate(entries):
                for right in entries[i + 1:]:
                    if (left[2] == right[2]
                            and max(left[0], right[0]) <= min(left[1], right[1])
                            and left[3] != right[3]):
                        conflict = True
            if conflict:
                warnings.append(f"Knob conflict for {env_var}; omitted")
                continue
            joined = ",".join(
                [f"{_fmt_bytes(start)}-{_fmt_bytes(end)}:{mem}:{value}"
                 for start, end, mem, value, _, _, _ in sorted(set(entries))]
                + [metadata.default])
            knob_env[env_var] = joined
            continue
        # A scalar is eligible only for one cell and full measured domain;
        # otherwise setting it globally leaks into an unsupported regime.
        values = {entry[3] for entry in entries}
        if (len(artifact_cells) == 1 and len(entries) == 1 and len(values) == 1
                and entries[0][0] == entries[0][5]
                and entries[0][1] == entries[0][6]):
            knob_env[env_var] = entries[0][3]
        else:
            warnings.append(f"Scalar knob {env_var} is not fully cell-scoped; omitted")

    return knob_env, warnings


def emit_conf(
    output_dir: Path,
    results: list,   # list[SweepResult]
    fingerprint: Fingerprint,
    *,
    accepted: bool = False,
    correctness: Optional[dict] = None,
    validation_points: Optional[list] = None,
) -> dict:           # output paths
    """
    Write a clearly labelled provisional artifact unless all acceptance gates
    and explicit default/tuned correctness evidence have passed.
    Returns a dict of {name: Path} for each output file.
    """
    correctness_ok = bool(
        correctness
        and correctness.get("default_complete")
        and correctness.get("tuned_complete")
        and not correctness.get("default_failures")
        and not correctness.get("new_tuned_failures")
    )
    validation_ok = _validation_covers_results(results, validation_points or [])
    if accepted and (not correctness_ok or not validation_ok):
        raise ValueError("accepted config requires passing paired validation and correctness evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    tune_tokens = _collect_tune_tokens(results)
    knob_env, knob_warnings = _collect_knob_overrides(results)

    label = "accepted" if accepted else "provisional-not-for-deployment"
    conf_lines = [f"# Status: {label}"] + _build_conf_lines(tune_tokens, knob_env, fingerprint)
    sh_lines = [f"# Status: {label}"] + _build_sh_lines(tune_tokens, knob_env, fingerprint)

    stem = "ucc_tuned" if accepted else "ucc_tuned_provisional"
    conf_path = output_dir / f"{stem}.conf"
    sh_path   = output_dir / f"{stem}_env.sh"
    fp_path   = output_dir / "fingerprint.json"
    res_path  = output_dir / "results.json"

    conf_path.write_text("\n".join(conf_lines) + "\n")
    sh_path.write_text("\n".join(sh_lines) + "\n")
    fp_path.write_text(json.dumps(dataclasses.asdict(fingerprint), indent=2))
    res_path.write_text(json.dumps({
        "status": label,
        "correctness": correctness,
        "validation": [
            {**{field.name: getattr(point, field.name)
                for field in dataclasses.fields(point) if field.name != "evidence"},
             "cell_key": (dataclasses.asdict(_point_cell_key(point))
                          if _point_cell_key(point) is not None else None),
             "evidence": point.evidence.to_dict() if point.evidence else None}
            for point in validation_points or []
        ],
        "results": _results_to_json(results),
    }, indent=2, sort_keys=True))

    if knob_warnings:
        (output_dir / "knob_conflicts.txt").write_text(
            "\n".join(knob_warnings) + "\n"
        )

    return {
        "conf": conf_path,
        "sh": sh_path,
        "fingerprint": fp_path,
        "results": res_path,
    }


def _build_conf_lines(
    tune_tokens: dict,
    knob_env: dict,
    fingerprint: Fingerprint,
) -> list:  # list[str]
    """Build lines for the UCC_CONFIG_FILE-compatible ucc_tuned.conf file."""
    lines = [
        "# UCC Tuning Configuration",
        f"# Generated : {fingerprint.timestamp}",
        f"# UCC       : {fingerprint.ucc_version}",
        f"# CPU       : {fingerprint.cpu_model}",
        f"# GPU       : {fingerprint.gpu_model}",
        f"# Driver    : {fingerprint.gpu_driver}",
        f"# Hash      : {fingerprint.hash}",
        "#",
        "# Set UCC_CONFIG_FILE=/path/to/ucc_tuned.conf to apply.",
        "# Correctness note: validate buffer correctness separately —",
        "#   ucc_perftest does not check output buffers.",
        "# Version note: long TUNE strings require UCC_INI_MAX_LINE >= 8192",
        "#   (UCC commit 281a0eb5 or newer); older UCC silently truncates.",
        "",
    ]
    for tune_var, tokens in sorted(tune_tokens.items()):
        lines.append(f"{tune_var}={'#'.join(tokens)}")
    if knob_env:
        lines.append("")
    for env_var, val in sorted(knob_env.items()):
        lines.append(f"{env_var}={val}")
    return lines


def _build_sh_lines(
    tune_tokens: dict,
    knob_env: dict,
    fingerprint: Fingerprint,
) -> list:  # list[str]
    """Build lines for the sourceable shell env script."""
    lines = [
        "#!/bin/sh",
        "# UCC Tuning Environment",
        f"# Generated : {fingerprint.timestamp}",
        f"# UCC       : {fingerprint.ucc_version}",
        f"# Hash      : {fingerprint.hash}",
        "#",
        "# Usage: source ucc_tuned_env.sh",
        "# Version note: long TUNE strings require UCC_INI_MAX_LINE >= 8192",
        "#   (UCC commit 281a0eb5 or newer); older UCC silently truncates.",
        "",
    ]
    for tune_var, tokens in sorted(tune_tokens.items()):
        lines.append(f"export {tune_var}='{('#').join(tokens)}'")
    for env_var, val in sorted(knob_env.items()):
        lines.append(f"export {env_var}='{val}'")
    return lines


def _results_to_json(results: list) -> list:
    """Serialize SweepResults to a JSON-safe list."""
    out = []
    for r in results:
        spec = r.spec
        bands = _compute_team_bands(spec.all_team_sizes or [spec.team_size])
        team_low, team_high = bands.get(spec.team_size, (spec.team_size, None))
        out.append({
            "cell_key": dataclasses.asdict(_spec_cell_key(spec)),
            "component":  spec.component,
            "collective": spec.collective,
            "mem_type":   spec.mem_type,
            "team_size":  spec.team_size,
            "requested_team_size": spec.team_size,
            "executed_team_size": spec.executed_team_size,
            "launcher": list(spec.mpi_launcher),
            "size_decisions": [
                {
                    "size_bytes":        d.size_bytes,
                    "should_override":   d.should_override,
                    "winner_name":       d.winner_name,
                    "winner_id":         d.winner_id,
                    "winner_median_us":  d.winner_median_us,
                    "default_median_us": d.default_median_us,
                    "margin":            d.margin,
                    "knob_overrides":    d.knob_overrides,
                    "policy":            d.policy.value,
                    "actual_size_bytes": d.actual_size_bytes,
                    "source":            d.source,
                    "paired_evidence":   d.evidence.to_dict() if d.evidence else None,
                    "knob_hypotheses":   [h.to_dict() for h in d.knob_hypotheses],
                    "default_selected_alg":       d.default_selected_alg,
                    "default_selected_component": d.default_selected_component,
                    "winner_cv":                  d.winner_cv,
                    "default_cv":                 d.default_cv,
                }
                for d in r.size_decisions
            ],
            "tune_ranges": [
                {
                    "start_bytes":    tr.start_bytes,
                    "end_bytes":      tr.end_bytes,
                    "alg_name":       tr.alg_name,
                    "alg_id":         tr.alg_id,
                    "knob_overrides": tr.knob_overrides,
                    "inclusive": True,
                    "resolution_bytes": tr.resolution_bytes,
                    "tune_token": tr.tune_token(
                        spec.collective,
                        _mem_type_for_tune(spec.mem_type),
                        team_low, team_high,
                    ),
                }
                for tr in r.tune_ranges
            ],
            "warnings": r.warnings,
            "proof_budget": dataclasses.asdict(r.proof_budget) if r.proof_budget else None,
            "unsupported_regimes": list(r.unsupported_regimes),
        })
    return out


# ---------------------------------------------------------------------------
# Stage 4 — validation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ValidationProbe:
    size_bytes: int
    inside: bool
    reasons: tuple[str, ...]


@dataclasses.dataclass
class ValidationPoint:
    collective: str
    mem_type: str
    size_bytes: int
    tuned_median_us: Optional[float]
    default_median_us: Optional[float]
    speedup: Optional[float]
    passed: bool
    inside: bool = True
    policy_selected: bool = True
    evidence: Optional[PairedEvidence] = None
    reason: str = ""
    component: Optional[str] = None
    team_size: Optional[int] = None
    requested_team_size: Optional[int] = None
    executed_team_size: Optional[int] = None
    datatype: str = "float32"
    reduction_op: str = "sum"


def validate(
    results: list,          # list[SweepResult] — the sweep output
    tune_tokens: dict,      # tune_var → list[token] from _collect_tune_tokens()
    knob_env: dict,         # env_var → val from _collect_knob_overrides()
    margin_threshold: float = 0.05,
    n_reps: int = 10,
    n_iter: int = 200,
    n_warmup: int = 20,
) -> list:    # list[ValidationPoint]
    """
    Compare the exact emitted configuration with the unmodified environment.
    This performance gate does not substitute for the separate correctness gate.
    """
    # Build the full tuned env: all tune vars + knob vars.
    tuned_env_base: dict = {}
    for tune_var, tokens in tune_tokens.items():
        tuned_env_base[tune_var] = "#".join(tokens)
    tuned_env_base.update(knob_env)

    points: list[ValidationPoint] = []

    for result in results:
        if not result.tune_ranges:
            continue
        spec = result.spec
        # Validate only what can reach the artifact.  _collect_tune_tokens()
        # drops zero-width ranges (UCC's score-map grammar rejects start>=end),
        # so probing one would demand that the emitted configuration select a
        # policy it was never handed, and report the guaranteed miss as a leak.
        emittable = [tr for tr in result.tune_ranges if tr.is_emittable]
        if not emittable:
            continue
        probes = _validation_probe_sizes(
            emittable, spec.msg_sizes_bytes,
            datatype=spec.datatype,
            resolution_bytes=spec.boundary_resolution_bytes,
            align_bytes=score_map_key_quantum(
                spec.collective, spec.datatype,
                spec.executed_team_size or spec.team_size or 1),
        )

        for probe in probes:
            size = probe.size_bytes
            count = collective_count_plan(
                spec.collective, spec.datatype, size,
                spec.executed_team_size or spec.team_size or 1).count
            # Validate what actually ships.  The default arm runs UCC
            # unmodified (ambient environment); the tuned arm runs exactly the
            # variables emit_conf() writes.  competition_env() isolation is a
            # sweep-time measurement aid that is absent from the artifact, so
            # injecting it here would certify a configuration nobody deploys
            # (e.g. an inf score that forces this component over a faster
            # global default would still pass).
            tuned_env = dict(tuned_env_base)
            default_env: dict = {}

            rs_tuned = RunSpec(
                collective=spec.collective,
                mem_type=spec.mem_type,
                count=count,
                datatype=spec.datatype,
                reduction_op=spec.reduction_op,
                n_reps=n_reps,
                n_iter=n_iter,
                n_warmup=n_warmup,
                persistent=spec.persistent,
                extra_env=tuned_env,
                mpi_launcher=list(spec.mpi_launcher),
                requested_team_size=spec.team_size,
                executed_team_size=spec.executed_team_size,
                perftest_path=spec.perftest_path,
                timeout_s=spec.timeout_s,
            )
            rs_default = dataclasses.replace(rs_tuned, extra_env=default_env)

            paired = measure_paired(
                rs_default, rs_tuned, seed=spec.confirmation_seed + 100000 + size,
                min_pairs=spec.min_pairs, max_pairs=spec.max_pairs,
                min_speedup=margin_threshold,
            )
            evidence = paired.evidence
            default_times = [s.latency_us for s in evidence.samples
                             if s.arm == "D" and s.ok and s.latency_us is not None]
            tuned_times = [s.latency_us for s in evidence.samples
                           if s.arm == "A" and s.ok and s.latency_us is not None]
            default_median = statistics.median(default_times) if default_times else None
            tuned_median = statistics.median(tuned_times) if tuned_times else None
            speedup = ((default_median - tuned_median) / default_median
                       if default_median and tuned_median is not None else None)
            # Compare against the ranges the artifact can actually carry: a
            # dropped zero-width range is not a policy the configuration leaked.
            selected = any(tr.contains(size) for tr in emittable)
            if probe.inside:
                # An inside probe asserts the emitted policy is a win at that
                # size.  ci_high <= 1.0 alone also passes a DEFAULT verdict --
                # e.g. a ratio interval of 0.96..0.99, which classify_evidence
                # rejected as inside the margin gate -- so "not a regression"
                # would certify sizes where the policy was never proven faster.
                passed = bool(evidence.decision == Decision.WIN
                              and evidence.ci_high is not None
                              and evidence.ci_high <= 1.0 and selected)
                reason = ("inside point confirmed a win" if passed
                          else "inside point is not a classified win")
            else:
                # The exact inclusive range model must select no algorithm at
                # an outside probe. Ranged knobs are checked by the same bounds
                # during construction in _collect_knob_overrides.
                statistically_identical = bool(
                    evidence.decision == Decision.DEFAULT
                    and evidence.ci_low is not None and evidence.ci_high is not None
                    and evidence.ci_low <= 1.0 <= evidence.ci_high
                )
                passed = not selected and statistically_identical
                reason = ("outside policy absent and behavior indistinguishable"
                          if passed else "outside behavior differs or policy leaked")
            vp = ValidationPoint(
                collective=spec.collective,
                mem_type=spec.mem_type,
                size_bytes=size,
                tuned_median_us=tuned_median,
                default_median_us=default_median,
                speedup=speedup,
                passed=passed,
                inside=probe.inside,
                policy_selected=selected,
                evidence=evidence,
                reason=reason,
                component=spec.component,
                team_size=spec.team_size,
                requested_team_size=spec.team_size,
                executed_team_size=spec.executed_team_size,
                datatype=spec.datatype,
                reduction_op=spec.reduction_op,
            )
            points.append(vp)

            status = "PASS" if passed else "FAIL"
            logger.info(
                "Validation [%s] %s/%s size=%s: tuned=%.2f us  default=%.2f us  speedup=%.1f%%",
                status, spec.collective, spec.mem_type, _fmt_bytes(size),
                tuned_median or float("nan"), default_median or float("nan"),
                (speedup or 0.0) * 100,
            )

    return points


def _validation_probe_sizes(
    tune_ranges: list, all_sizes: list, *, datatype: str = "float32",
    resolution_bytes: int = 1024, align_bytes: Optional[int] = None,
) -> list[ValidationProbe]:
    """Return aligned inside/outside boundary, anchor, midpoint, and quartile probes."""
    if not all_sizes:
        return []
    # Probes step on the score-map key grid, not merely the element grid: for a
    # team-scaled collective two sizes one element apart are the *same* UCC
    # entry, so an element-stepped "just-outside" probe would re-measure the
    # boundary it is supposed to sit outside of.
    alignment = dtype_size(datatype) if align_bytes is None else align_bytes

    def floor_unit(value: int) -> int:
        return max(alignment, (value // alignment) * alignment)

    domain_low = floor_unit(min(all_sizes))
    domain_high = floor_unit(max(all_sizes))
    collected: dict[tuple[int, bool], set[str]] = {}

    def add(size: int, inside: bool, reason: str) -> None:
        aligned = floor_unit(size)
        if domain_low <= aligned <= domain_high:
            collected.setdefault((aligned, inside), set()).add(reason)

    for tune_range in tune_ranges:
        start, end = tune_range.start_bytes, tune_range.end_bytes
        add(start, True, "start")
        add(end, True, "inclusive-end")
        if start + alignment <= end:
            add(start + alignment, True, "just-inside-start")
            add(end - alignment, True, "just-inside-end")
        add(start - alignment, False, "just-outside-start")
        add(end + alignment, False, "just-outside-end")
        width = end - start
        if width > 2 * resolution_bytes:
            add(start + width // 2, True, "midpoint")
            add(start + width // 4, True, "quartile-25")
            add(start + (3 * width) // 4, True, "quartile-75")
        for anchor in all_sizes:
            aligned = floor_unit(anchor)
            if start <= aligned <= end:
                add(aligned, True, "original-anchor")
        for decision in tune_range.evidence_points:
            anchor = getattr(decision, "actual_size_bytes", None)
            if anchor is not None:
                add(anchor, True, "refined-anchor")
    # At a boundary shared by adjacent ranges, the exact config legitimately
    # selects the neighboring policy. Validate it as an inside point once,
    # rather than also demanding that the whole config be default there.
    for key in list(collected):
        size, inside = key
        if not inside and any(tune_range.contains(size) for tune_range in tune_ranges):
            del collected[key]
    return [ValidationProbe(size, inside, tuple(sorted(reasons)))
            for (size, inside), reasons in sorted(collected.items())]


def _representative_sizes(tune_ranges: list, all_sizes: list) -> list:
    """Compatibility alias: all deterministic validation probe byte sizes."""
    return [probe.size_bytes for probe in _validation_probe_sizes(tune_ranges, all_sizes)]


def trim_failed_ranges(results: list, validation_points: list,
                       trim_round: int) -> list:
    """Trim to passing confirmed neighbors; after two rounds remove failures."""
    if trim_round < 1 or trim_round > 2:
        raise ValueError("at most two trim/regenerate rounds are permitted")
    by_cell: dict[CellKey, list[ValidationPoint]] = {}
    for point in validation_points:
        key = _point_cell_key(point)
        if key is not None:
            by_cell.setdefault(key, []).append(point)
    for result in results:
        points = by_cell.get(_spec_cell_key(result.spec), [])
        retained = []
        for tune_range in result.tune_ranges:
            inside = [p for p in points if p.inside and tune_range.contains(p.size_bytes)]
            passed = sorted(p.size_bytes for p in inside if p.passed)
            if inside and len(passed) != len(inside):
                if trim_round == 2 or not passed:
                    continue
                tune_range.start_bytes = min(passed)
                tune_range.end_bytes = max(passed)
            retained.append(tune_range)
        result.tune_ranges = retained
    return results


def _validation_covers_results(results: list, points: list) -> bool:
    """Require a passing final-config result for every deterministic probe."""
    if not results or not points:
        return False
    for result in results:
        if not result.tune_ranges:
            continue
        expected = _validation_probe_sizes(
            result.tune_ranges, result.spec.msg_sizes_bytes,
            datatype=result.spec.datatype,
            resolution_bytes=result.spec.boundary_resolution_bytes,
        )
        for probe in expected:
            matches = [
                point for point in points
                if _point_cell_key(point) == _spec_cell_key(result.spec)
                and point.size_bytes == probe.size_bytes
                and point.inside == probe.inside
            ]
            if not matches or not all(point.passed for point in matches):
                return False
    return True


def validate_with_trimming(
    results: list,
    *,
    margin_threshold: float = 0.05,
    n_reps: int = 10,
    n_iter: int = 200,
    n_warmup: int = 20,
    max_confirmation_points: int = 40,
    used_confirmation_points: int = 0,
) -> list:
    """Validate, then trim/regenerate at most twice without relaxing a gate."""
    points: list[ValidationPoint] = []
    used_points = used_confirmation_points
    for round_number in range(3):
        required_points = sum(
            len(_validation_probe_sizes(
                result.tune_ranges, result.spec.msg_sizes_bytes,
                datatype=result.spec.datatype,
                resolution_bytes=result.spec.boundary_resolution_bytes,
            ))
            for result in results
        )
        if used_points + required_points > max_confirmation_points:
            for result in results:
                if result.tune_ranges:
                    result.warnings.append(
                        "final validation point budget exhausted; ranges omitted")
                    result.tune_ranges = []
            return []
        tune_tokens = _collect_tune_tokens(results)
        knob_env, _ = _collect_knob_overrides(results)
        points = validate(
            results, tune_tokens, knob_env,
            margin_threshold=margin_threshold, n_reps=n_reps,
            n_iter=n_iter, n_warmup=n_warmup,
        )
        used_points += len(points)
        if points and all(point.passed for point in points):
            break
        if round_number == 2:
            # No third trim/regenerate cycle is permitted.
            for result in results:
                result.tune_ranges = []
            break
        trim_failed_ranges(results, points, round_number + 1)
        if not any(result.tune_ranges for result in results):
            break
    return points

def validate_screening(
    results: list,
    *,
    margin_threshold: float = 0.05,
    n_reps: int = 7,
    n_iter: int = 1000,
    n_warmup: int = 100,
) -> list:    # list[ValidationPoint]
    """Screening-mode Stage 4: median comparison of config vs default.

    Cheaper than the paired `validate()`: measures each range endpoint with
    `measure()` (no paired confirmation) and requires the median speedup to
    clear the margin.  Used by the default fast path.
    """
    tune_tokens = _collect_tune_tokens(results)
    knob_env, _ = _collect_knob_overrides(results)
    tuned_env_base: dict = {}
    for tune_var, tokens in tune_tokens.items():
        tuned_env_base[tune_var] = "#".join(tokens)
    tuned_env_base.update(knob_env)

    def measure_or_none(rs: RunSpec):
        try:
            return measure(rs)
        except RuntimeError as exc:
            logger.warning("screening validation measurement failed: %s", exc)
            return None

    points: list[ValidationPoint] = []
    for result in results:
        if not result.tune_ranges:
            continue
        spec = result.spec
        # Screening compares the emitted configuration against unmodified UCC,
        # matching validate(); isolation vars are not part of the artifact.
        for tr in result.tune_ranges:
            if not tr.is_emittable:
                # Never emitted, so there is no policy here to verify; probing it
                # would compare the artifact against a token UCC never received.
                continue
            for size in sorted({tr.start_bytes, tr.end_bytes}):
                count = collective_count_plan(
                    spec.collective, spec.datatype, size,
                    spec.executed_team_size or spec.team_size or 1).count
                rs_tuned = RunSpec(
                    collective=spec.collective, mem_type=spec.mem_type,
                    count=count, datatype=spec.datatype,
                    reduction_op=spec.reduction_op,
                    n_reps=n_reps, n_iter=n_iter, n_warmup=n_warmup,
                    persistent=spec.persistent,
                    extra_env=dict(tuned_env_base),
                    mpi_launcher=list(spec.mpi_launcher),
                    requested_team_size=spec.team_size,
                    executed_team_size=spec.executed_team_size,
                    perftest_path=spec.perftest_path,
                    timeout_s=spec.timeout_s,
                )
                rs_default = dataclasses.replace(rs_tuned, extra_env={})
                tuned_result = measure_or_none(rs_tuned)
                default_result = measure_or_none(rs_default)
                default_median = default_result.median_us if default_result else None
                tuned_median = tuned_result.median_us if tuned_result else None
                if (default_median is None or tuned_median is None
                        or default_median <= 0):
                    passed, speedup, reason = False, None, "measurement failed or non-positive default"
                else:
                    speedup = (default_median - tuned_median) / default_median
                    passed = speedup > margin_threshold
                    reason = ("screening median speedup above margin"
                              if passed else "screening median speedup within margin")
                points.append(ValidationPoint(
                    collective=spec.collective, mem_type=spec.mem_type,
                    size_bytes=size, tuned_median_us=tuned_median,
                    default_median_us=default_median, speedup=speedup,
                    passed=passed, inside=True, policy_selected=True,
                    evidence=None, reason=reason,
                    component=spec.component, team_size=spec.team_size,
                    requested_team_size=spec.team_size,
                    executed_team_size=spec.executed_team_size,
                    datatype=spec.datatype, reduction_op=spec.reduction_op,
                ))
    return points



# ---------------------------------------------------------------------------
# Tuning summary log
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    results: list,
    validation_points: list,
    fingerprint: Fingerprint,
    skipped: list,          # list[str] describing skipped cells
) -> Path:
    """Write a human-readable tuning_summary.txt."""
    lines = [
        "=" * 72,
        "UCC Offline Tuner — Tuning Summary",
        "=" * 72,
        "",
        fingerprint.summary(),
        "",
        "-" * 72,
        "Tuning results",
        "-" * 72,
    ]

    for result in results:
        spec = result.spec
        label = (f"{spec.component}/{spec.collective} "
                  f"mem={spec.mem_type} team_size={spec.team_size} "
                  f"datatype={spec.datatype} op={spec.reduction_op}")
        lines.append(f"\n{label}")
        if not result.size_decisions:
            # No size was measured at all — every algorithm failed at every
            # size. That is a broken run, not a verdict about UCC's default.
            # Reported on hpcac-internal job 10464, where a UCX transport
            # fault made every perftest launch abort and all four cells still
            # read "UCC default is within margin".
            lines.append("  !! NO MEASUREMENTS — every algorithm failed at every "
                         "size; this cell was NOT tuned")
            for w in result.warnings:
                lines.append(f"     {w}")
            continue
        if not result.tune_ranges:
            lines.append("  (no overrides needed — UCC default is within margin)")
            continue
        bands = _compute_team_bands(spec.all_team_sizes or [spec.team_size])
        team_low, team_high = bands.get(spec.team_size, (spec.team_size, None))
        for tr in result.tune_ranges:
            mt = _mem_type_for_tune(spec.mem_type)
            tok = tr.tune_token(spec.collective, mt, team_low, team_high)
            lines.append(f"  {tok}")
            if tr.knob_overrides:
                for k, v in sorted(tr.knob_overrides.items()):
                    lines.append(f"    {k}={v}")
            lines.append(
                f"    inclusive coverage [{tr.start_bytes}, {tr.end_bytes}], "
                f"sampled resolution={tr.resolution_bytes} B"
            )
        overridden = sum(1 for d in result.size_decisions if d.should_override)
        total = len(result.size_decisions)
        lines.append(f"  ({overridden}/{total} size points overridden)")
        if result.proof_budget:
            budget = result.proof_budget
            lines.append(
                f"  proof budget: points={budget.used_points}/{budget.max_points}, "
                f"pairs={budget.used_pairs} (per-point max={budget.max_pairs})"
            )
        for decision in result.size_decisions:
            reason = decision.evidence.reason if decision.evidence else decision.source
            lines.append(
                f"  evidence size={decision.actual_size_bytes}: "
                f"{decision.policy.value} ({reason})"
            )
        if result.warnings:
            for w in result.warnings:
                lines.append(f"  WARNING: {w}")

    if skipped:
        lines += ["", "-" * 72, "Skipped cells (no algorithms or all sizes failed)", "-" * 72]
        for s in skipped:
            lines.append(f"  {s}")

    if validation_points:
        lines += ["", "-" * 72, "Stage 4 validation", "-" * 72]
        pass_count = sum(1 for v in validation_points if v.passed)
        lines.append(f"  {pass_count}/{len(validation_points)} points passed")
        for vp in validation_points:
            status = "PASS" if vp.passed else "FAIL"
            lines.append(
                f"  [{status}] {vp.component}/{vp.collective}/{vp.mem_type} "
                f"team={vp.requested_team_size} datatype={vp.datatype} "
                f"op={vp.reduction_op} "
                f"size={_fmt_bytes(vp.size_bytes)}: "
                f"speedup={(vp.speedup or 0.0)*100:.1f}% "
                f"(tuned={(vp.tuned_median_us or float('nan')):.1f}us "
                f"default={(vp.default_median_us or float('nan')):.1f}us; "
                f"{vp.reason})"
            )
        lines.append("")
        lines.append(
            "STATUS: PROVISIONAL. Correctness evidence is unavailable; no "
            "accepted/deployable config may be written."
        )

    summary_path = output_dir / "tuning_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def _compute_cell_budget(
    total_cells: int,
    done_cells: int,
    max_points: int,
    used_points: int,
    per_cell_min: int,
) -> int:
    """Per-cell confirmation budget with a floor guarantee.

    Each cell is guaranteed at least *per_cell_min* points (or the entire
    remaining pool if that is smaller).  The budget for the current cell is
    computed by reserving *per_cell_min* for every future cell and splitting
    any leftover surplus equally across the remaining cells.
    """
    remaining = total_cells - done_cells
    if remaining <= 0:
        return 0
    pool = max_points - used_points
    if pool <= 0:
        return 0
    reserved = remaining * per_cell_min
    surplus = max(0, pool - reserved)
    per_cell = per_cell_min + surplus // remaining
    return min(per_cell, pool)


# ---------------------------------------------------------------------------
# Cost model (ROADMAP A2)
# ---------------------------------------------------------------------------

def compute_cost_model(
    pairs: list,               # [(component, collective), ...]
    alg_map: dict,
    mem_types: list,
    team_sizes: list,
    sizes: list,
    n_reps: int,
    skip_asymmetric: bool = True,
) -> dict:
    """Compute the planned screening invocation count (ROADMAP §3).

    Per size the screening path measures every advertised algorithm plus the
    default arm, each ``n_reps`` independent perftest launches:

        invocations = cells x sizes x (n_algs + 1) x n_reps

    Knob work, boundary refinement, and Stage-4 validation are not part of the
    screening path and are excluded (validation is data-dependent).
    """
    cells: list[dict] = []
    total = 0
    skipped_cells = 0
    for comp, coll in pairs:
        if is_unkeyable_collective(coll):
            # UCC cannot derive the score-map key for these (--force-asymmetric
            # cannot change that), so no byte range can express or validate the
            # workload; measuring them would only bill time for inert policy.
            skipped_cells += len(mem_types) * len(team_sizes)
            continue
        if skip_asymmetric and coll in _ASYMMETRIC_COLLS:
            skipped_cells += len(mem_types) * len(team_sizes)
            continue
        n_algs = len(alg_map.get(comp, {}).get(coll, []))
        for mem_type in mem_types:
            for team_size in team_sizes:
                invocations = len(sizes) * (n_algs + 1) * n_reps
                cells.append({
                    "component": comp, "collective": coll,
                    "mem_type": mem_type, "team_size": team_size,
                    "n_algs": n_algs, "invocations": invocations,
                })
                total += invocations
    return {
        "cells": len(cells),
        "skipped_cells": skipped_cells,
        "sizes": len(sizes),
        "n_reps": n_reps,
        "total_invocations": total,
        "cells_detail": cells,
    }


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def print_cost_model(model: dict, launch_overhead: float) -> None:
    """Print the planned invocation count and estimated wall-clock."""
    total = model["total_invocations"]
    wall = total * launch_overhead
    print("Planned sweep cost (screening path)")
    print("=" * 60)
    print(f"cells:            {model['cells']}")
    print(f"skipped cells:    {model['skipped_cells']}")
    print(f"sizes per cell:   {model['sizes']}")
    print(f"repetitions:      {model['n_reps']}")
    print(f"total invocations: {total}")
    print(f"est. wall-clock:  {_fmt_duration(wall)} "
          f"(at {launch_overhead:g} s/launch)")
    print("  (Stage-4 validation and knob work excluded; the latter is "
          "data-dependent.)")
    print()
    print("per-cell breakdown:")
    for cell in model["cells_detail"]:
        print(f"  {cell['component']}/{cell['collective']} "
              f"mem={cell['mem_type']} team={cell['team_size']}: "
              f"{cell['n_algs']} algs -> {cell['invocations']} invocations")


# ---------------------------------------------------------------------------
# Findings (ROADMAP A3)
# ---------------------------------------------------------------------------

def _build_findings(results: list) -> list:
    """Build the ranked lead list from sweep results (ROADMAP A3).

    Each emitted message range is one lead: its cell, inclusive byte range,
    UCC's default-selected algorithm (from A1 readback), the winning
    algorithm, the median screening margin, the number of measured points, and
    the winning arm's CV.  Sorted by opportunity: speedup, then coverage.
    """
    leads: list[dict] = []
    for result in results:
        spec = result.spec
        cell = dataclasses.asdict(_spec_cell_key(spec))
        for tr in result.tune_ranges:
            points = tr.evidence_points
            margins = [d.margin for d in points if d.margin and d.margin > 0]
            cvs = [d.winner_cv for d in points if d.winner_cv is not None]
            default_algs = [d.default_selected_alg
                            for d in points if d.default_selected_alg]
            default_comps = [d.default_selected_component
                             for d in points if d.default_selected_component]
            leads.append({
                "cell_key": cell,
                "component": spec.component,
                "collective": spec.collective,
                "mem_type": spec.mem_type,
                "team_size": spec.team_size,
                "message_range": {
                    "start_bytes": tr.start_bytes,
                    "end_bytes": tr.end_bytes,
                },
                "default_algorithm": default_algs[0] if default_algs else None,
                "default_component": default_comps[0] if default_comps else None,
                "winning_algorithm": tr.alg_name,
                "speedup": statistics.median(margins) if margins else 0.0,
                "n_points": len(points),
                "cv": statistics.median(cvs) if cvs else None,
                "coverage_bytes": tr.end_bytes - tr.start_bytes + 1,
            })
    leads.sort(key=lambda lead: (
        -lead["speedup"], -lead["coverage_bytes"],
        lead["collective"], lead["team_size"],
    ))
    for rank, lead in enumerate(leads, 1):
        lead["rank"] = rank
    return leads


def write_findings(
    output_dir: Path,
    results: list,
    fingerprint: Fingerprint,
) -> dict:
    """Write findings.json + findings.md: ranked leads for human triage."""
    leads = _build_findings(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "findings.json"
    md_path = output_dir / "findings.md"

    json_path.write_text(json.dumps({
        "status": "provisional-leads-not-for-deployment",
        "generated": fingerprint.timestamp,
        "fingerprint": dataclasses.asdict(fingerprint),
        "leads": leads,
    }, indent=2, sort_keys=True))

    lines = [
        "# UCC Offline Tuner — Findings",
        "",
        "Status: **leads for human triage — NOT a deployable config.**",
        "",
        fingerprint.summary(),
        "",
        "## Ranked leads",
        "",
    ]
    if not leads:
        lines.append("(none — no cell emitted a message-range override)")
    for lead in leads:
        rng = lead["message_range"]
        default = lead["default_algorithm"] or "(unknown — use --readback-level debug)"
        cv = f"{lead['cv'] * 100:.1f}%" if lead["cv"] is not None else "n/a"
        lines.append(
            f"{lead['rank']}. {lead['component']}/{lead['collective']} "
            f"mem={lead['mem_type']} team={lead['team_size']} "
            f"[{_fmt_bytes(rng['start_bytes'])}, {_fmt_bytes(rng['end_bytes'])}]: "
            f"default={default} -> {lead['winning_algorithm']}, "
            f"speedup={lead['speedup'] * 100:.1f}%, "
            f"n={lead['n_points']}, CV={cv}"
        )
    lines += [
        "",
        "## Legend",
        "",
        "- speedup: median screening margin across the range's measured points",
        "- n: number of measured size points in the range",
        "- CV: median coefficient-of-variation of the winning algorithm's timing",
        "- default: algorithm UCC's default arm selected (score-map readback;",
        "  empty unless run with --readback-level debug)",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": json_path, "md": md_path}


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_tuning(
    component_collective_pairs: list,   # [(component, collective), ...]
    mem_types: list,                    # ["host", "cuda", ...]
    team_sizes: list,                   # [8, 64, ...]
    msg_sizes_bytes: list,
    skip_asymmetric: bool = True,
    alg_map: Optional[dict] = None,
    datatype: str = "float32",
    reduction_op: str = "sum",
    n_reps: int = 7,
    n_iter: int = 1000,
    n_warmup: int = 100,
    persistent: bool = True,
    margin_threshold: float = 0.05,
    min_pairs: int = 10,
    max_pairs: int = 20,
    boundary_resolution_bytes: int = 1024,
    max_boundary_probes: int = 4,
    max_confirmation_points: int = 40,
    per_cell_min_points: int = 10,
    proof_mode: bool = False,
    skip_knobs: bool = False,
    confirmation_seed: int = 0,
    mpi_launcher: Optional[list] = None,
    perftest_path: str = "ucc_perftest",
    ucc_info_path: str = "ucc_info",
    timeout_s: int = 120,
    extra_ucc_info_env: Optional[dict] = None,
    readback_log_level: Optional[str] = "info",
) -> tuple[list, list]:   # (results, skipped_messages)
    """
    Stage 1–3: enumerate algorithms, sweep all cells, return SweepResults.

    Returns (results: list[SweepResult], skipped: list[str]).
    Skipped cells are logged so the output doesn't overclaim coverage.
    """
    if mpi_launcher is None:
        mpi_launcher = ["mpirun", "-np", "1"]

    logger.info("Stage 1: enumerating algorithms via ucc_info -A")
    raw_output = ""
    if alg_map is None:
        raw_output = run_ucc_info_raw(ucc_info_path, extra_ucc_info_env or {})
        alg_map = parse_ucc_info_algs(raw_output)
    if not alg_map:
        head = " ".join(raw_output.split("\n")[:5]) if raw_output else "(no output)"
        raise RuntimeError(
            f"ucc_info ({ucc_info_path}) returned empty algorithm map. "
            f"Output preview: {head}"
        )

    results: list[SweepResult] = []
    skipped: list[str] = []
    confirmation_points_used = 0

    total = len(component_collective_pairs) * len(mem_types) * len(team_sizes)
    done = 0

    for comp, coll in component_collective_pairs:
        if is_unkeyable_collective(coll):
            msg = (f"{comp}/{coll}: skipped collective with no locally derivable "
                   "score-map size (ucc_coll_args_msgsize returns "
                   "UCC_MSG_SIZE_ASYMMETRIC); byte-range tuning cannot express it")
            logger.warning(msg)
            skipped.append(msg)
            done += len(mem_types) * len(team_sizes)
            continue
        if skip_asymmetric and coll in _ASYMMETRIC_COLLS:
            msg = (f"{comp}/{coll}: skipped asymmetric collective "
                   "(use --force-asymmetric to include)")
            logger.warning(msg)
            skipped.append(msg)
            done += len(mem_types) * len(team_sizes)
            continue

        alg_list = alg_map.get(comp, {}).get(coll, [])
        if not alg_list:
            msg = (f"{comp}/{coll}: no algorithms found in ucc_info -A "
                   f"(component not built or collective not supported)")
            logger.warning(msg)
            skipped.append(msg)
            done += len(mem_types) * len(team_sizes)
            continue

        for mem_type in mem_types:
            for team_size in team_sizes:
                bound_launcher, executed_team_size = bind_launcher_team_size(
                    mpi_launcher, team_size)
                logger.info(
                    "Stage 2/3 [%d/%d]: %s/%s mem=%s team_size=%d",
                    done + 1, total, comp, coll, mem_type, team_size,
                )
                spec = SweepSpec(
                    component=comp,
                    collective=coll,
                    mem_type=mem_type,
                    team_size=team_size,
                    msg_sizes_bytes=list(msg_sizes_bytes),
                    all_team_sizes=team_sizes,
                    alg_list=list(alg_list),
                    datatype=datatype,
                    reduction_op=reduction_op,
                    n_reps=n_reps,
                    n_iter=n_iter,
                    n_warmup=n_warmup,
                    persistent=persistent,
                    margin_threshold=margin_threshold,
                    min_pairs=min_pairs,
                    max_pairs=max_pairs,
                    boundary_resolution_bytes=boundary_resolution_bytes,
                    max_boundary_probes=max_boundary_probes,
                    max_confirmation_points=_compute_cell_budget(
                        total, done, max_confirmation_points,
                        confirmation_points_used, per_cell_min_points),
                    mpi_launcher=bound_launcher,
                    executed_team_size=executed_team_size,
                    perftest_path=perftest_path,
                    timeout_s=timeout_s,
                    proof_mode=proof_mode,
                    skip_knobs=skip_knobs,
                    confirmation_seed=confirmation_seed,
                    readback_log_level=readback_log_level,
                )
                done += 1
                result = sweep_cell(spec)
                results.append(result)
                if result.proof_budget:
                    confirmation_points_used += result.proof_budget.used_points

                if not result.size_decisions:
                    # Distinguish "measured, default was good enough" from
                    # "nothing measured". Collapsing the two makes a totally
                    # failed run look like a clean no-op result.
                    msg = (f"{comp}/{coll} mem={mem_type} team_size={team_size}: "
                           "NO MEASUREMENTS — every algorithm failed at every size; "
                           "cell was not tuned")
                    logger.error(msg)
                    skipped.append(msg)
                elif not result.tune_ranges:
                    msg = (f"{comp}/{coll} mem={mem_type} team_size={team_size}: "
                           "UCC default is within margin for all sizes — no override emitted")
                    skipped.append(msg)

    return results, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UCC offline algorithm tuner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--collective", default="allreduce",
        help="Comma-separated collective names to tune.",
    )
    p.add_argument(
        "--component", default="",
        help=(
            "Comma-separated components to sweep (e.g. tl/ucp,tl/cuda). "
            "Default: all components that have the requested collective in ucc_info -A."
        ),
    )
    p.add_argument(
        "--mem-type", default="host",
        help="Comma-separated memory types (host, cuda, cuda-mng).",
    )
    p.add_argument(
        "--team-sizes", default="8",
        help="Comma-separated team sizes to tune for.",
    )
    p.add_argument("--min-bytes", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=1 << 20,
                   help="Max message size in bytes (default 1 MiB).")
    p.add_argument("--factor", type=int, default=2,
                   help="Grid multiplication factor (2 or 4).")
    p.add_argument("--datatype", default="float32")
    p.add_argument("--op", default="sum", dest="reduction_op")
    p.add_argument("--n-reps", type=int, default=7,
                   help="Independent perftest repetitions per measurement.")
    p.add_argument("--n-iter", type=int, default=1000,
                   help="Perftest -n iterations per rep.")
    p.add_argument("--n-warmup", type=int, default=100,
                   help="Perftest -w warmup iterations per rep.")
    p.add_argument("--no-persistent", action="store_true",
                   help="Disable persistent mode (includes init/finalize overhead).")
    p.add_argument("--min-speedup", "--margin", type=float, default=0.05,
                   dest="min_speedup",
                   help="Required speedup for an override to be emitted.")
    p.add_argument("--min-pairs", type=int, default=10)
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--boundary-resolution-bytes", type=int, default=1024)
    p.add_argument("--max-boundary-probes", type=int, default=4)
    p.add_argument("--max-confirmation-points", type=int, default=40)
    p.add_argument("--per-cell-min-points", type=int, default=10,
                   help="Minimum confirmation points guaranteed per cell.")
    p.add_argument("--proof-mode", action="store_true",
                   help=("Run full paired confirmation, boundary refinement, and "
                         "knob attribution (expensive). Default is fast screening only."))
    p.add_argument("--no-knobs", action="store_true",
                   help="In --proof-mode, skip knob attribution (algorithm-only).")
    p.add_argument("--seed", type=int, default=0,
                   help="Recorded seed for balanced paired order.")
    p.add_argument("--readback-level", default="info",
                   choices=["off", "info", "debug"],
                   help=("Score-map readback (ROADMAP A1): parse which "
                         "component/algorithm UCC selected per arm. 'info' "
                         "records the component; 'debug' additionally records "
                         "the algorithm via the init-function symbol; 'off' "
                         "disables readback."))
    p.add_argument("--launcher", default="mpirun -np {team_size}",
                   help=("Launcher prefix with exactly one rank option. Its integer "
                         "value is rebound per cell; {team_size} is recommended."))
    p.add_argument("--perftest", default="ucc_perftest")
    p.add_argument("--ucc-info", default="ucc_info")
    p.add_argument("--ucx-info", default="ucx_info")
    p.add_argument("--output-dir", default="./ucc_tuning_output",
                   help="Directory to write output files.")
    p.add_argument("--no-validate", action="store_true",
                    help="Skip Stage 4 validation runs.")
    p.add_argument("--force-asymmetric", action="store_true",
                   help=("Include asymmetric/*v/rooted collectives in the sweep "
                         "(default: skip them as tuning is dubious)."))
    p.add_argument("--dry-run", action="store_true",
                   help=("Print the planned invocation count and estimated "
                         "wall-clock (ROADMAP A2), then exit without launching "
                         "any perftest measurement."))
    p.add_argument("--launch-overhead", type=float, default=3.0,
                   help=("Estimated per-launch wall-clock seconds (mpirun/srun "
                         "launch + perftest init) used for the --dry-run "
                         "wall-clock estimate."))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not 0 < args.min_speedup < 1:
        parser.error("--min-speedup must be between 0 and 1")
    if args.min_pairs < 10 or args.max_pairs < args.min_pairs or args.max_pairs > 20:
        parser.error("pair limits must satisfy 10 <= min-pairs <= max-pairs <= 20")
    if args.boundary_resolution_bytes <= 0:
        parser.error("--boundary-resolution-bytes must be positive")
    if not 0 <= args.max_boundary_probes <= 12:
        parser.error("--max-boundary-probes must be between 0 and 12")
    if args.max_confirmation_points <= 0:
        parser.error("--max-confirmation-points must be positive")

    collectives = [c.strip() for c in args.collective.split(",")]
    sizeless = [c for c in collectives if is_sizeless_collective(c)]
    if sizeless:
        parser.error(
            "--collective without a message-size dimension (perftest reports "
            f"N/A) cannot be range-tuned: {', '.join(sizeless)}")
    if args.min_bytes < 1:
        parser.error("--min-bytes must be positive")
    if args.max_bytes < args.min_bytes:
        parser.error("--max-bytes must be >= --min-bytes")
    boundary_resolution = 256 if args.proof_mode else args.boundary_resolution_bytes
    boundary_probes = 12 if args.proof_mode else args.max_boundary_probes
    confirmation_points = (
        max(80, args.max_confirmation_points)
        if args.proof_mode else args.max_confirmation_points
    )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    try:
        mpi_launcher = shlex.split(args.launcher)
        team_sizes = [int(t.strip()) for t in args.team_sizes.split(",")]
        if not team_sizes or any(size <= 0 for size in team_sizes):
            raise ValueError("team sizes must be positive")
        for team_size in team_sizes:
            bind_launcher_team_size(mpi_launcher, team_size)
    except ValueError as exc:
        parser.error(str(exc))

    # Stage 0: fingerprint
    logger.info("Stage 0: collecting platform fingerprint")
    fingerprint = collect_fingerprint(args.ucc_info, args.ucx_info)
    logger.info("Fingerprint:\n%s", fingerprint.summary())

    # Build search space from CLI args
    mem_types   = [m.strip() for m in args.mem_type.split(",")]
    sizes       = msg_size_grid(args.min_bytes, args.max_bytes, args.factor)

    # Resolve component/collective pairs
    alg_map = run_ucc_info_algs(args.ucc_info)
    if args.component:
        components = [c.strip() for c in args.component.split(",")]
        pairs = [(comp, coll) for comp in components for coll in collectives]
    else:
        # Auto-discover: use any component that has the requested collective.
        pairs = [
            (comp, coll)
            for coll in collectives
            for comp in alg_map
            if coll in alg_map.get(comp, {})
        ]
        if not pairs:
            logger.error(
                "No components found with collectives %s. "
                "Run ucc_info -A to see what is available.",
                collectives,
            )
            return 1

    logger.info(
        "Tuning %d (component, collective) × %d mem_types × %d team_sizes "
        "= %d cells, %d sizes each",
        len(pairs), len(mem_types), len(team_sizes),
        len(pairs) * len(mem_types) * len(team_sizes),
        len(sizes),
    )

    if args.dry_run:
        model = compute_cost_model(
            pairs, alg_map, mem_types, team_sizes, sizes, args.n_reps,
            skip_asymmetric=not args.force_asymmetric,
        )
        print_cost_model(model, args.launch_overhead)
        return 0

    # Stages 1–3: sweep (alg_map passed to avoid duplicate ucc_info call)
    results, skipped = run_tuning(
        component_collective_pairs=pairs,
        mem_types=mem_types,
        team_sizes=team_sizes,
        msg_sizes_bytes=sizes,
        skip_asymmetric=not args.force_asymmetric,
        alg_map=alg_map,
        datatype=args.datatype,
        reduction_op=args.reduction_op,
        n_reps=args.n_reps,
        n_iter=args.n_iter,
        n_warmup=args.n_warmup,
        persistent=not args.no_persistent,
        margin_threshold=args.min_speedup,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        boundary_resolution_bytes=boundary_resolution,
        max_boundary_probes=boundary_probes,
        max_confirmation_points=confirmation_points,
        per_cell_min_points=args.per_cell_min_points,
        mpi_launcher=mpi_launcher,
        perftest_path=args.perftest,
        ucc_info_path=args.ucc_info,
        timeout_s=300,
        proof_mode=args.proof_mode,
        skip_knobs=args.no_knobs,
        confirmation_seed=args.seed,
        readback_log_level=None if args.readback_level == "off" else args.readback_level,
    )

    # Stage 4: validate
    validation_points: list = []
    if not args.no_validate:
        logger.info("Stage 4: validating generated config")
        if args.proof_mode:
            validation_points = validate_with_trimming(
                results,
                margin_threshold=args.min_speedup,
                n_reps=max(args.min_pairs, args.n_reps),
                n_iter=args.n_iter // 5,
                n_warmup=args.n_warmup // 5,
                max_confirmation_points=confirmation_points,
                used_confirmation_points=sum(
                    result.proof_budget.used_points
                    for result in results if result.proof_budget
                ),
            )
        else:
            validation_points = validate_screening(
                results,
                margin_threshold=args.min_speedup,
                n_reps=args.n_reps,
                n_iter=args.n_iter,
                n_warmup=args.n_warmup,
            )
        fail_count = sum(1 for v in validation_points if not v.passed)
        if fail_count:
            logger.warning(
                "%d/%d validation points failed — review tuning_summary.txt",
                fail_count, len(validation_points),
            )

    # Emit output files
    logger.info("Emitting output files to %s", output_dir)
    paths = emit_conf(output_dir, results, fingerprint)
    summary_path = write_summary(
        output_dir, results, validation_points, fingerprint, skipped
    )
    paths["summary"] = summary_path
    findings_paths = write_findings(output_dir, results, fingerprint)
    paths.update(findings_paths)

    logger.info("Done.")
    logger.info("  Config  : %s", paths["conf"])
    logger.info("  Shell   : %s", paths["sh"])
    logger.info("  Summary : %s", paths["summary"])
    logger.info("  Findings: %s", paths["md"])
    logger.info("")
    logger.info(
        "PROVISIONAL ONLY: no accepted config was written because this local "
        "run has no supplied default/tuned correctness evidence."
    )

    # Cells whose launches all failed (transport, launcher, or perftest) are a
    # failed run, not a clean "default was already best" no-op: such a cell
    # has no ranges and therefore no validation points to fail on.
    unmeasured = [result for result in results if not result.size_decisions]
    if unmeasured:
        logger.error(
            "%d/%d swept cells produced no measurements — nothing was tuned; "
            "see tuning_summary.txt for the per-cell failures",
            len(unmeasured), len(results))

    fail_validation = any(not v.passed for v in validation_points)
    return 1 if (fail_validation or unmeasured) else 0


if __name__ == "__main__":
    sys.exit(main())
