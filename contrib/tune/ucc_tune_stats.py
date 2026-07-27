#!/usr/bin/env python3
"""Deterministic evidence and statistics for conservative UCC tuning.

Screening measurements deliberately do not enter this module.  A decision in
this module is based on complete, temporally adjacent pairs and is safe-by-
default: malformed, incomplete, noisy, or order-confounded evidence is
classified as DEFAULT.
"""

from __future__ import annotations

import bisect
import dataclasses
import enum
import hashlib
import itertools
import math
import random
import statistics
from typing import Iterable, Optional


class Decision(str, enum.Enum):
    WIN = "WIN"
    DEFAULT = "DEFAULT"
    REGRESSION = "REGRESSION"


@dataclasses.dataclass(frozen=True, order=True)
class CellKey:
    component: str
    collective: str
    mem_type: str
    team_size: int
    datatype: str = "float32"
    op: str = "sum"


@dataclasses.dataclass(frozen=True)
class ArmSample:
    pair_id: int
    order: str
    arm: str
    latency_us: Optional[float]
    ok: bool = True
    failure: Optional[str] = None
    seed: int = 0
    size_bytes: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class PairedEvidence:
    samples: tuple[ArmSample, ...]
    complete_pairs: int
    ratio: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    adjusted_alpha: float
    order_p: Optional[float]
    decision: Decision
    reason: str
    p_win: Optional[float] = None
    p_regression: Optional[float] = None
    cv_default: Optional[float] = None
    cv_candidate: Optional[float] = None
    log_ratio_iqr: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a stable, JSON-ready representation with samples in run order."""
        return {
            "samples": [dataclasses.asdict(s) for s in self.samples],
            "complete_pairs": self.complete_pairs,
            "ratio": self.ratio,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "adjusted_alpha": self.adjusted_alpha,
            "order_p": self.order_p,
            "decision": self.decision.value,
            "reason": self.reason,
            "p_win": self.p_win,
            "p_regression": self.p_regression,
            "cv_default": self.cv_default,
            "cv_candidate": self.cv_candidate,
            "log_ratio_iqr": self.log_ratio_iqr,
        }


@dataclasses.dataclass(frozen=True)
class PointDecision:
    size_bytes: int
    policy: Decision
    candidate: Optional[str]
    algorithm_evidence: Optional[PairedEvidence] = None
    knob_evidence: tuple[PairedEvidence, ...] = ()
    correctness: Optional[bool] = None
    source: str = "paired-confirmation"
    actual_size_bytes: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class InclusiveTuneRange:
    start_bytes: int
    end_bytes: int
    cell_key: CellKey
    candidate: str
    evidence_points: tuple[PointDecision, ...]
    resolution_bytes: int
    knob_ranges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.start_bytes < 0 or self.end_bytes < self.start_bytes:
            raise ValueError("invalid inclusive range")

    def contains(self, size_bytes: int) -> bool:
        return self.start_bytes <= size_bytes <= self.end_bytes


@dataclasses.dataclass(frozen=True)
class ProofBudget:
    max_points: int = 40
    max_pairs: int = 20
    used_points: int = 0
    used_pairs: int = 0

    def __post_init__(self) -> None:
        if self.max_points < 0 or self.max_pairs <= 0:
            raise ValueError("proof point limit must be non-negative and pair limit positive")
        if not (0 <= self.used_points <= self.max_points):
            raise ValueError("used_points exceeds proof budget")
        if self.used_pairs < 0:
            raise ValueError("used_pairs cannot be negative")

    def consume(self, pairs: int) -> "ProofBudget":
        if self.used_points >= self.max_points:
            raise RuntimeError("confirmation point budget exhausted")
        return dataclasses.replace(
            self, used_points=self.used_points + 1,
            used_pairs=self.used_pairs + pairs,
        )


# Sign-flip nulls are enumerated exhaustively up to 2**_EXACT_SIGN_LIMIT
# assignments.  Beyond that the proof budget allows 2**20 ~ 1.0M sign
# combinations per null (and classify_cell evaluates every group twice), so
# the tail uses a seeded PRNG whose seed is derived from the exact sample
# values: identical inputs always reproduce identical p-values and intervals.
_EXACT_SIGN_LIMIT = 13
_SIGN_FLIP_SAMPLES = 40_001


def _sign_flip_null(centered: list[float]) -> list[float]:
    """Null distribution of the mean under all 2**n sign flips of `centered`.

    Exhaustive through 2**_EXACT_SIGN_LIMIT assignments; deterministic PRNG
    sampling beyond (seed derived from the exact sample values).
    """
    n = len(centered)
    if (1 << n) <= (1 << _EXACT_SIGN_LIMIT):
        return [
            statistics.fmean(sign * value for sign, value in zip(signs, centered))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        ]
    digest = hashlib.sha256(repr(tuple(centered)).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    bitmask = (1 << n) - 1
    values = []
    for _ in range(_SIGN_FLIP_SAMPLES):
        bits = rng.getrandbits(64) & bitmask
        values.append(sum(
            (value if bits >> index & 1 else -value)
            for index, value in enumerate(centered)) / n)
    return values


def _quantile(values: list[float], probability: float) -> float:
    """Conservative nearest-rank quantile, deterministic for small samples."""
    if not values:
        raise ValueError("quantile of empty data")
    ordered = sorted(values)
    rank = math.ceil(probability * len(ordered)) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return math.inf if mean <= 0 else statistics.stdev(values) / mean


def _complete_pairs(
    samples: Iterable[ArmSample], default_arm: str, candidate_arm: str,
) -> tuple[list[tuple[str, float, float]], Optional[str]]:
    grouped: dict[int, dict[str, ArmSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.pair_id, {})[sample.arm] = sample
    pairs: list[tuple[str, float, float]] = []
    for pair_id in sorted(grouped):
        arms = grouped[pair_id]
        d = arms.get(default_arm)
        c = arms.get(candidate_arm)
        if d is None:
            return pairs, "missing default arm"
        if c is None:
            return pairs, "missing candidate arm"
        if not d.ok:
            return pairs, "failed default arm"
        if not c.ok:
            return pairs, "failed candidate arm"
        if (d.latency_us is None or c.latency_us is None
                or not math.isfinite(d.latency_us)
                or not math.isfinite(c.latency_us)
                or d.latency_us <= 0 or c.latency_us <= 0):
            return pairs, "non-finite or non-positive timing"
        if d.order != c.order or d.order not in ("AB", "BA"):
            return pairs, "invalid order stratum"
        pairs.append((d.order, d.latency_us, c.latency_us))
    return pairs, None


def _exact_centered_interval(log_ratios: list[float], alpha: float) -> tuple[float, float]:
    """Sign-flip interval for the mean; sampled beyond the exact limit."""
    mean = statistics.fmean(log_ratios)
    residuals = [x - mean for x in log_ratios]
    permutations = _sign_flip_null(residuals)
    lo_noise = _quantile(permutations, alpha / 2)
    hi_noise = _quantile(permutations, 1 - alpha / 2)
    return mean - hi_noise, mean - lo_noise


def _one_sided_p(log_ratios: list[float], null: float, side: str) -> float:
    centered = [x - null for x in log_ratios]
    observed = statistics.fmean(centered)
    values = sorted(_sign_flip_null(centered))
    if side == "lower":
        hits = bisect.bisect_right(values, observed + 1e-15)
    else:
        hits = len(values) - bisect.bisect_left(values, observed - 1e-15)
    return hits / len(values)


def _order_interaction(log_ratios: list[float], orders: list[str]) -> tuple[float, bool]:
    ab = [x for x, order in zip(log_ratios, orders) if order == "AB"]
    ba = [x for x, order in zip(log_ratios, orders) if order == "BA"]
    if not ab or not ba:
        return 0.0, True
    observed = abs(statistics.fmean(ab) - statistics.fmean(ba))
    n_ab = len(ab)
    differences = []
    indexes = range(len(log_ratios))
    for selected in itertools.combinations(indexes, n_ab):
        selected_set = set(selected)
        left = [x for i, x in enumerate(log_ratios) if i in selected_set]
        right = [x for i, x in enumerate(log_ratios) if i not in selected_set]
        differences.append(abs(statistics.fmean(left) - statistics.fmean(right)))
    p_value = sum(v >= observed - 1e-15 for v in differences) / len(differences)
    return p_value, False


def paired_log_ratio_ci(
    samples: Iterable[ArmSample], *, alpha: float = 0.05,
    default_arm: str = "D", candidate_arm: str = "A",
) -> tuple[float, float, float]:
    """Return geometric mean ratio and exact two-sided sign-flip interval."""
    pairs, error = _complete_pairs(samples, default_arm, candidate_arm)
    if error:
        raise ValueError(error)
    if not pairs:
        raise ValueError("no complete pairs")
    ratios = [math.log(candidate / default) for _, default, candidate in pairs]
    lo, hi = _exact_centered_interval(ratios, alpha)
    return math.exp(statistics.fmean(ratios)), math.exp(lo), math.exp(hi)


def classify_evidence(
    samples: Iterable[ArmSample], *, min_pairs: int = 10,
    min_speedup: float = 0.05, adjusted_alpha: float = 0.05,
    cv_threshold: float = 0.10, require_stable_variance: bool = True,
    log_iqr_threshold: float = 0.10,
    partial_sweep: bool = False, default_arm: str = "D",
    candidate_arm: str = "A",
) -> PairedEvidence:
    """Classify paired evidence; every incomplete/unsafe state falls back."""
    frozen_samples = tuple(samples)
    if not (0 < min_speedup < 1):
        raise ValueError("min_speedup must be between 0 and 1")
    if not (0 < adjusted_alpha < 1):
        raise ValueError("adjusted_alpha must be between 0 and 1")
    pairs, error = _complete_pairs(frozen_samples, default_arm, candidate_arm)

    def rejected(reason: str) -> PairedEvidence:
        return PairedEvidence(frozen_samples, len(pairs), None, None, None,
                              adjusted_alpha, None, Decision.DEFAULT, reason)

    if partial_sweep:
        return rejected("partial algorithm sweep")
    if error:
        return rejected(error)
    if len(pairs) < min_pairs:
        return rejected(f"insufficient complete pairs: {len(pairs)} < {min_pairs}")

    orders = [order for order, _, _ in pairs]
    defaults = [default for _, default, _ in pairs]
    candidates = [candidate for _, _, candidate in pairs]
    logs = [math.log(candidate / default) for _, default, candidate in pairs]
    cv_default, cv_candidate = _cv(defaults), _cv(candidates)
    ordered_logs = sorted(logs)
    log_iqr = (ordered_logs[(3 * len(logs)) // 4] - ordered_logs[len(logs) // 4]
               if len(logs) >= 4 else 0.0)
    ratio, ci_low, ci_high = paired_log_ratio_ci(
        frozen_samples, alpha=adjusted_alpha,
        default_arm=default_arm, candidate_arm=candidate_arm,
    )
    order_p, missing_order = _order_interaction(logs, orders)
    p_win = _one_sided_p(logs, math.log(1 - min_speedup), "lower")
    p_regression = _one_sided_p(logs, 0.0, "upper")

    decision = Decision.DEFAULT
    reason = "confidence interval does not prove a safe material win"
    if missing_order:
        reason = "missing AB/BA order stratum"
    elif order_p < adjusted_alpha:
        reason = "significant order interaction"
    elif require_stable_variance and (cv_default > cv_threshold or cv_candidate > cv_threshold
                                      or log_iqr > log_iqr_threshold):
        reason = "persistent CV or paired-log spread above threshold"
    elif ci_low > 1.0:
        decision, reason = Decision.REGRESSION, "lower confidence bound exceeds 1.0"
    elif abs(ratio - 1.0) <= 1e-15:
        reason = "numerical tie"
    elif (statistics.fmean([x for x, o in zip(logs, orders) if o == "AB"]) >= 0
          or statistics.fmean([x for x, o in zip(logs, orders) if o == "BA"]) >= 0):
        reason = "AB/BA strata are not both on the winning side"
    elif ci_high < 1 - min_speedup:
        decision, reason = Decision.WIN, "upper confidence bound proves material win"

    return PairedEvidence(
        frozen_samples, len(pairs), ratio, ci_low, ci_high, adjusted_alpha,
        order_p, decision, reason, p_win, p_regression,
        cv_default, cv_candidate, log_iqr,
    )


def holm_adjusted_alphas(p_values: Iterable[float], alpha: float = 0.05) -> tuple[float, ...]:
    """Return per-hypothesis Holm step-down alpha values in input order."""
    values = tuple(p_values)
    if not values:
        return ()
    ranked = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    for rank, (index, _) in enumerate(ranked):
        result[index] = alpha / (len(values) - rank)
    return tuple(result)


def classify_cell(
    sample_groups: Iterable[Iterable[ArmSample]], *,
    hypothesis_ids: Optional[Iterable[str]] = None,
    arm_pairs: Optional[Iterable[tuple[str, str]]] = None, **kwargs,
) -> tuple[PairedEvidence, ...]:
    """Classify all decisions in a cell with deterministic Holm correction.

    ``arm_pairs`` names the (default, candidate) arm labels of each group when
    they are not the shared ``default_arm``/``candidate_arm``.  A proof-mode
    cell mixes plain D/A comparisons with knob-attribution comparisons that
    measure the same two conditions under the A0/A1 labels; classifying those
    samples against D/A finds neither arm and reports every one of them as
    "missing default arm", so no knob hypothesis can survive attribution.
    """
    groups = tuple(tuple(group) for group in sample_groups)
    ids = tuple(hypothesis_ids) if hypothesis_ids is not None else tuple(
        f"hypothesis-{index:08d}" for index in range(len(groups)))
    if len(ids) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError("hypothesis_ids must be unique and match sample_groups")
    shared_default = kwargs.pop("default_arm", "D")
    shared_alt = kwargs.pop("candidate_arm", "A")
    if arm_pairs is None:
        pairs = ((shared_default, shared_alt),) * len(groups)
    else:
        pairs = tuple(tuple(pair) for pair in arm_pairs)
        if len(pairs) != len(groups):
            raise ValueError("arm_pairs must match sample_groups")
        for pair in pairs:
            if len(pair) != 2 or not all(pair):
                raise ValueError("arm_pairs must be pairs of arm names")
            if pair[0] == pair[1]:
                raise ValueError("arm pair labels must differ")

    def classify(group: tuple[ArmSample, ...], pair: tuple[str, str],
                 **over) -> PairedEvidence:
        return classify_evidence(group, default_arm=pair[0],
                                 candidate_arm=pair[1], **over)

    preliminary = tuple(classify(group, pair, **kwargs)
                        for group, pair in zip(groups, pairs))
    p_values = [e.p_win if e.p_win is not None else 1.0 for e in preliminary]
    base_alpha = float(kwargs.get("adjusted_alpha", 0.05))
    ranked = sorted(range(len(preliminary)),
                     key=lambda index: (p_values[index], ids[index]))
    alphas_list = [0.0] * len(preliminary)
    for rank, index in enumerate(ranked):
        alphas_list[index] = base_alpha / (len(preliminary) - rank)
    alphas = tuple(alphas_list)
    adjusted_kwargs = dict(kwargs)
    adjusted_kwargs.pop("adjusted_alpha", None)
    classified = [
        classify(group, pair, adjusted_alpha=cell_alpha, **adjusted_kwargs)
        for group, pair, cell_alpha in zip(groups, pairs, alphas)
    ]
    # Holm is step-down: after the first ordered hypothesis misses its gate,
    # no later hypothesis in that family can be accepted as a win.
    gate_open = True
    for rank, index in enumerate(ranked):
        gate = base_alpha / (len(preliminary) - rank)
        if p_values[index] > gate:
            gate_open = False
        if not gate_open and classified[index].decision == Decision.WIN:
            classified[index] = dataclasses.replace(
                classified[index], decision=Decision.DEFAULT,
                reason="Holm step-down gate not passed",
            )
    return tuple(classified)
