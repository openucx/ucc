#!/usr/bin/env python3
"""Exact deterministic proof fixtures for paired UCC tuning statistics."""

import dataclasses
import math
import unittest

from ucc_tune_stats import (
    ArmSample, CellKey, Decision, InclusiveTuneRange, PointDecision,
    ProofBudget, classify_cell, classify_evidence, holm_adjusted_alphas,
    paired_log_ratio_ci,
)


def samples(ratios, *, orders=None, defaults=None):
    orders = orders or ["AB" if i % 2 == 0 else "BA" for i in range(len(ratios))]
    defaults = defaults or [100.0] * len(ratios)
    result = []
    for pair_id, (ratio, order, default) in enumerate(zip(ratios, orders, defaults)):
        result.extend((ArmSample(pair_id, order, "D", default),
                       ArmSample(pair_id, order, "A", default * ratio)))
    return result


class TestPairedDecisions(unittest.TestCase):
    def test_material_win(self):
        evidence = classify_evidence(samples([.80] * 10))
        self.assertEqual(evidence.decision, Decision.WIN)
        self.assertLess(evidence.ci_high, .95)

    def test_numerical_tie(self):
        evidence = classify_evidence(samples([1.0] * 10))
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertEqual(evidence.reason, "numerical tie")

    def test_within_noise(self):
        evidence = classify_evidence(samples([.90, .99, .94, .97, .91] * 2),
                                     cv_threshold=1)
        self.assertEqual(evidence.decision, Decision.DEFAULT)

    def test_supported_regression(self):
        evidence = classify_evidence(samples([1.126] * 10))
        self.assertEqual(evidence.decision, Decision.REGRESSION)
        self.assertGreater(evidence.ci_low, 1)

    def test_exact_interval_is_deterministic(self):
        first = paired_log_ratio_ci(samples([.8, .82] * 5))
        second = paired_log_ratio_ci(samples([.8, .82] * 5))
        self.assertEqual(first, second)

    def test_sampled_interval_is_deterministic_beyond_exact_limit(self):
        # 20 pairs exceeds the exact sign-flip enumeration cap; the sampled
        # path must stay reproducible and keep its bounds consistent.
        first = paired_log_ratio_ci(samples([.82, .86, .90, .94] * 5))
        second = paired_log_ratio_ci(samples([.82, .86, .90, .94] * 5))
        self.assertEqual(first, second)
        ratio, ci_low, ci_high = first
        self.assertLess(ci_low, ratio)
        self.assertLess(ratio, ci_high)
        self.assertLess(ci_high, 1.0)

    def test_nine_pairs(self):
        evidence = classify_evidence(samples([.5] * 9))
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertIn("9 < 10", evidence.reason)

    def test_missing_default(self):
        evidence = classify_evidence([ArmSample(0, "AB", "A", 1)] * 10)
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertIn("missing default", evidence.reason)

    def test_failed_candidate(self):
        fixture = samples([.8] * 10)
        fixture[1] = dataclasses.replace(fixture[1], ok=False, failure="timeout")
        self.assertEqual(classify_evidence(fixture).decision, Decision.DEFAULT)

    def test_non_finite(self):
        fixture = samples([.8] * 10)
        fixture[1] = dataclasses.replace(fixture[1], latency_us=math.nan)
        self.assertEqual(classify_evidence(fixture).decision, Decision.DEFAULT)

    def test_high_variance(self):
        evidence = classify_evidence(samples([.8] * 10, defaults=[50, 150] * 5))
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertIn("CV", evidence.reason)

    def test_missing_order_stratum(self):
        evidence = classify_evidence(samples([.8] * 10, orders=["AB"] * 10))
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertIn("missing AB/BA", evidence.reason)

    def test_ab_ba_drift(self):
        ratios = [.8 if i % 2 == 0 else 1.0 for i in range(10)]
        evidence = classify_evidence(samples(ratios), cv_threshold=1)
        self.assertEqual(evidence.decision, Decision.DEFAULT)
        self.assertIn("order", evidence.reason)

    def test_partial_algorithm_failure(self):
        evidence = classify_evidence(samples([.5] * 10), partial_sweep=True)
        self.assertEqual(evidence.decision, Decision.DEFAULT)

    def test_holm_adjustment(self):
        self.assertEqual(holm_adjusted_alphas([.01, .04, .02]),
                         (.05 / 3, .05, .025))
        decisions = classify_cell([samples([.8] * 10), samples([.8] * 10)])
        self.assertEqual([e.adjusted_alpha for e in decisions], [.025, .05])

    def test_raw_knob_wins_can_fail_complete_family(self):
        marginal = samples([.8] * 6 + [1.0] * 4)
        self.assertEqual(classify_evidence(marginal, cv_threshold=1,
                                           log_iqr_threshold=1).decision,
                         Decision.WIN)
        corrected = classify_cell(
            [marginal] * 3,
            hypothesis_ids=["knob:K:4:algorithm", "knob:K:4:joint",
                            "knob:K:4:knob-effect"],
            cv_threshold=1, log_iqr_threshold=1,
        )
        self.assertTrue(all(e.decision == Decision.DEFAULT for e in corrected))
        self.assertEqual(corrected[0].adjusted_alpha, .05 / 3)
        self.assertTrue(any(e.reason == "Holm step-down gate not passed"
                            for e in corrected))

    def test_all_knob_gates_legitimately_survive(self):
        strong = samples([.8] * 10)
        corrected = classify_cell(
            [strong] * 3, hypothesis_ids=["a", "b", "c"])
        self.assertTrue(all(e.decision == Decision.WIN for e in corrected))

    def test_hypothesis_identity_makes_dictionary_order_irrelevant(self):
        groups = {"z": samples([.8] * 6 + [1.0] * 4),
                  "a": samples([.8] * 10),
                  "m": samples([.85] * 7 + [.99] * 3)}
        first_ids = tuple(groups)
        first = classify_cell([groups[key] for key in first_ids],
                              hypothesis_ids=first_ids, cv_threshold=1,
                              log_iqr_threshold=1)
        second_ids = tuple(reversed(first_ids))
        second = classify_cell([groups[key] for key in second_ids],
                               hypothesis_ids=second_ids, cv_threshold=1,
                               log_iqr_threshold=1)
        first_by_id = {key: (value.adjusted_alpha, value.decision, value.reason)
                       for key, value in zip(first_ids, first)}
        second_by_id = {key: (value.adjusted_alpha, value.decision, value.reason)
                        for key, value in zip(second_ids, second)}
        self.assertEqual(first_by_id, second_by_id)


class TestImmutableModel(unittest.TestCase):
    def test_records_are_frozen(self):
        key = CellKey("tl/ucp", "allreduce", "host", 8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            key.team_size = 64

    def test_inclusive_range(self):
        key = CellKey("tl/ucp", "allreduce", "host", 8)
        point = PointDecision(4096, Decision.WIN, "knomial")
        tune_range = InclusiveTuneRange(4096, 4100, key, "knomial", (point,), 4)
        self.assertTrue(tune_range.contains(4100))
        self.assertFalse(tune_range.contains(4101))

    def test_budget_is_bounded(self):
        budget = ProofBudget(1, 20).consume(10)
        with self.assertRaises(RuntimeError):
            budget.consume(10)


class TestRetainedClassificationFixtures(unittest.TestCase):
    """Task-8/88 summaries only; raw Gaia artifacts are not locally replayed."""

    def test_61763_seven_pair_allreduce_is_classification_only(self):
        win = classify_evidence(samples([.836] * 7), min_pairs=7)
        regression = classify_evidence(samples([1.126] * 7), min_pairs=7)
        self.assertEqual(win.decision, Decision.WIN)
        self.assertEqual(regression.decision, Decision.REGRESSION)
        self.assertEqual(classify_evidence(samples([.836] * 7)).decision,
                         Decision.DEFAULT)  # never emission proof

    def test_61763_supported_collective_fixtures(self):
        allgather = classify_evidence(samples([.75] * 7), min_pairs=7)
        reduce_scatter = classify_evidence(samples([.50] * 7), min_pairs=7)
        self.assertEqual((allgather.decision, reduce_scatter.decision),
                         (Decision.WIN, Decision.WIN))

    def test_61763_six_identical_alltoall_controls_emit_zero(self):
        controls = [classify_evidence(samples([1.0] * 7), min_pairs=7)
                    for _ in range(6)]
        self.assertEqual(sum(e.decision == Decision.WIN for e in controls), 0)

    def test_61761_is_order_confounded(self):
        evidence = classify_evidence(samples([.7] * 10, orders=["AB"] * 10))
        self.assertEqual(evidence.decision, Decision.DEFAULT)



def labeled_samples(ratios, arms=("D", "A")):
    """Paired samples under caller-chosen arm labels."""
    result = []
    for pair_id, ratio in enumerate(ratios):
        order = "AB" if pair_id % 2 == 0 else "BA"
        result.extend((ArmSample(pair_id, order, arms[0], 100.0),
                        ArmSample(pair_id, order, arms[1], 100.0 * ratio)))
    return result


class TestClassifyCellArmPairs(unittest.TestCase):
    """A cell mixes arm vocabularies: plain D/A and knob-attribution A0/A1.

    confirm_knob() measures its knob-effect gate as A1 vs A0 so the three
    attribution gates stay distinguishable.  Classifying those samples against
    the cell-wide D/A labels found neither arm, so every knob hypothesis was
    reclassified "missing default arm" and no knob could ever survive.
    """

    def test_each_group_uses_the_labels_it_was_measured_under(self):
        algorithm = labeled_samples([.80] * 10)
        knob_effect = labeled_samples([.80] * 10, arms=("A0", "A1"))
        outcomes = classify_cell(
            [algorithm, knob_effect],
            hypothesis_ids=("algorithm:1", "knob:knob-effect"),
            arm_pairs=(("D", "A"), ("A0", "A1")), min_pairs=10)
        self.assertEqual([outcome.decision for outcome in outcomes],
                         [Decision.WIN, Decision.WIN])
        self.assertEqual([outcome.reason for outcome in outcomes],
                         ["upper confidence bound proves material win",
                         "upper confidence bound proves material win"])

    def test_shared_labels_reject_foreign_arm_vocabularies(self):
        # The gate itself is not loosened: samples labelled A0/A1 classified
        # against D/A must still be refused, which is exactly what happened to
        # every knob hypothesis before per-group labels existed.
        outcomes = classify_cell(
            [labeled_samples([.80] * 10, arms=("A0", "A1"))],
            hypothesis_ids=("knob:knob-effect",), min_pairs=10)
        self.assertEqual(outcomes[0].decision, Decision.DEFAULT)
        self.assertEqual(outcomes[0].reason, "missing default arm")

    def test_shared_labels_can_be_overridden_for_the_whole_cell(self):
        outcomes = classify_cell(
            [labeled_samples([.80] * 10, arms=("X", "Y"))],
            hypothesis_ids=("screening",), min_pairs=10,
            default_arm="X", candidate_arm="Y")
        self.assertEqual(outcomes[0].decision, Decision.WIN)

    def test_holm_step_down_still_governs_mixed_labels(self):
        strong = labeled_samples([.50] * 10)
        marginal = labeled_samples([.985] * 10, arms=("A0", "A1"))
        outcomes = classify_cell(
            [strong, marginal], hypothesis_ids=("a", "b"),
            arm_pairs=(("D", "A"), ("A0", "A1")), min_pairs=10)
        self.assertEqual(outcomes[0].decision, Decision.WIN)
        self.assertNotEqual(outcomes[1].decision, Decision.WIN)

    def test_arm_pair_count_must_match_the_family(self):
        with self.assertRaisesRegex(ValueError, "arm_pairs must match"):
            classify_cell([labeled_samples([.8] * 10)] * 2,
                          arm_pairs=(("D", "A"),), min_pairs=10)

    def test_arm_pair_labels_must_be_distinct_and_present(self):
        group = labeled_samples([.8] * 10)
        with self.assertRaisesRegex(ValueError, "must differ"):
            classify_cell([group], arm_pairs=(("A", "A"),), min_pairs=10)
        with self.assertRaisesRegex(ValueError, "pairs of arm names"):
            classify_cell([group], arm_pairs=(("A",),), min_pairs=10)


if __name__ == "__main__":
    unittest.main()
