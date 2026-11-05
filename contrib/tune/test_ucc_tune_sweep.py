#!/usr/bin/env python3
"""Deterministic tests for conservative inclusive sweep behavior."""

import unittest
from unittest.mock import MagicMock, patch

from ucc_tune_runner import RunResult, SingleRunSample
from ucc_tune_space import AlgInfo, Knob
from ucc_tune_stats import ArmSample, Decision, ProofBudget, classify_evidence
from ucc_tune_sweep import (
    SizeDecision, SweepSpec, TuneRange, _compute_team_bands, _fmt_bytes,
    _forced_alg_env, _mem_type_for_tune, coalesce_ranges, confirm_knob,
    refine_boundaries, sweep_cell,
)


def _run(us, cv=0.01):
    sample = SingleRunSample(1024, 4096, us, us, us)
    return RunResult(MagicMock(), [sample], us, 0, cv, 1, 0, 0, cv > .1)


def _evidence(ratio=.8, n=10, decision=None, arms=("D", "A")):
    """Paired evidence with the arm labels the caller's measurement really used.

    confirm_knob() measures the knob-effect gate under the A0/A1 labels so the
    three attribution gates stay distinguishable; a fixture that labelled every
    gate D/A let cell-wide classification find "missing default arm" for the
    knob hypotheses while the tests still passed, so the labels are a parameter
    here rather than a constant.
    """
    samples = []
    for i in range(n):
        order = "AB" if i % 2 == 0 else "BA"
        samples.extend((ArmSample(i, order, arms[0], 100),
                        ArmSample(i, order, arms[1], 100 * ratio)))
    result = classify_evidence(samples, default_arm=arms[0], candidate_arm=arms[1])
    if decision is not None:
        result = __import__("dataclasses").replace(result, decision=decision)
    return result


def _spec(**kwargs):
    values = dict(component="tl/ucp", collective="allreduce", mem_type="host",
                  team_size=8, msg_sizes_bytes=[4096],
                  alg_list=[AlgInfo(0, "knomial", "")], n_reps=3)
    values.update(kwargs)
    return SweepSpec(**values)


def _decision(size, policy=Decision.WIN, alg="knomial", knobs=None):
    return SizeDecision(size, policy == Decision.WIN, alg, 0, 8, 10, .2,
                        knobs or {}, policy, _evidence(.8), size, "fixture")


class TestFormatting(unittest.TestCase):
    def test_exact_unit_formatting(self):
        self.assertEqual(_fmt_bytes(4096), "4k")
        self.assertEqual(_fmt_bytes(16380), "16380")
        self.assertEqual(_fmt_bytes(1 << 20), "1M")

    def test_memory_mapping(self):
        self.assertEqual(_mem_type_for_tune("cuda-mng"), "cuda_managed")
        self.assertNotEqual(_mem_type_for_tune("cuda-mng"), "cuda-managed")
        with self.assertRaises(ValueError):
            _mem_type_for_tune("unknown")

    def test_forced_environment_is_screening_only_broad(self):
        env = _forced_alg_env(_spec(), "knomial")
        self.assertIn("@knomial", env["UCC_TL_UCP_TUNE"])
        self.assertEqual(env["UCC_TLS"], "ucp")


class TestInclusiveRanges(unittest.TestCase):
    def test_finite_token_has_exact_end_and_team(self):
        tune_range = TuneRange(4096, 16380, "knomial", 0, {})
        self.assertEqual(
            tune_range.tune_token("allreduce", "host", 8),
            "allreduce:4k-16380:host:[8-8]:inf:@knomial",
        )

    def test_open_tail_forbidden(self):
        with self.assertRaises(ValueError):
            TuneRange(4096, None, "knomial", 0, {})

    def test_membership_is_inclusive(self):
        tune_range = TuneRange(4096, 16380, "knomial", 0, {})
        self.assertTrue(tune_range.contains(4096))
        self.assertTrue(tune_range.contains(16380))
        self.assertFalse(tune_range.contains(16384))

    def test_zero_width_range_is_not_emittable(self):
        self.assertTrue(TuneRange(4096, 4100, "knomial", 0, {}).is_emittable)
        self.assertFalse(TuneRange(4096, 4096, "knomial", 0, {}).is_emittable)

    def test_single_team_never_extrapolates(self):
        self.assertEqual(_compute_team_bands([8]), {8: (8, 8)})
        self.assertEqual(_compute_team_bands([64, 8]), {8: (8, 8), 64: (64, 64)})

    def test_default_anchor_breaks_range(self):
        decisions = [_decision(4096), _decision(16384, Decision.DEFAULT)]
        ranges = coalesce_ranges(decisions, [4096, 16384])
        self.assertEqual([(r.start_bytes, r.end_bytes) for r in ranges], [(4096, 4096)])

    def test_regression_anchor_never_inherited(self):
        decisions = [_decision(4096), _decision(4100),
                     _decision(16380, Decision.REGRESSION),
                     _decision(16384, Decision.DEFAULT)]
        ranges = coalesce_ranges(decisions, [4096, 4100, 16380, 16384])
        self.assertEqual((ranges[0].start_bytes, ranges[0].end_bytes), (4096, 4100))
        for size in (16380, 16384, 16388):
            self.assertFalse(any(r.contains(size) for r in ranges))

    def test_sparse_like_wins_remain_singletons(self):
        ranges = coalesce_ranges([_decision(4096), _decision(16380)],
                                 [4096, 16380], resolution_bytes=1024)
        self.assertEqual([(r.start_bytes, r.end_bytes) for r in ranges],
                         [(4096, 4096), (16380, 16380)])

    def test_resolved_like_wins_merge(self):
        ranges = coalesce_ranges([_decision(4096), _decision(4100)],
                                 [4096, 4100], resolution_bytes=1024)
        self.assertEqual([(r.start_bytes, r.end_bytes) for r in ranges], [(4096, 4100)])

    def test_knob_values_split(self):
        a = _decision(4096, knobs={"K": "2"})
        b = _decision(4100, knobs={"K": "4"})
        self.assertEqual(len(coalesce_ranges([a, b], [4096, 4100])), 2)


class TestRefinement(unittest.TestCase):
    def test_alignment_and_cap(self):
        calls = []
        def confirm(size, alg, alg_id):
            calls.append(size)
            return _decision(size, Decision.WIN if size < 8000 else Decision.DEFAULT)
        refined, budget = refine_boundaries(
            [_decision(4096), _decision(16384, Decision.DEFAULT)], confirm,
            datatype="float32", resolution_bytes=256, max_probes=2,
            budget=ProofBudget(40, 20),
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(size % 4 == 0 for size in calls))
        self.assertEqual(budget.used_points, 2)
        self.assertEqual(len({d.actual_size_bytes for d in refined}), len(refined))

    def test_point_budget_exhaustion_adds_no_probe(self):
        budget = ProofBudget(1, 20, 1, 10)
        refined, final = refine_boundaries(
            [_decision(4096), _decision(16384, Decision.DEFAULT)],
            lambda *args: self.fail("must not probe"), budget=budget,
        )
        self.assertEqual(len(refined), 2)
        self.assertEqual(final, budget)

    def test_multiple_transitions_are_considered(self):
        calls = []
        decisions = [_decision(1024), _decision(4096, Decision.DEFAULT),
                     _decision(8192)]
        refine_boundaries(decisions,
                          lambda size, alg, aid: calls.append(size) or _decision(size),
                          resolution_bytes=256, max_probes=1)
        self.assertEqual(len(calls), 2)


class TestAttribution(unittest.TestCase):
    @patch("ucc_tune_sweep._paired_compare")
    def test_three_arms_all_must_win(self, compare):
        compare.side_effect = [_evidence(.8), _evidence(.8), _evidence(.8)]
        knob = Knob("K", "", "auto", ("4",))
        value, evidence = confirm_knob(_spec(), 4096, "knomial", knob, "4")
        self.assertEqual(value, "4")
        self.assertEqual(len(evidence), 3)

    @patch("ucc_tune_sweep._paired_compare")
    def test_knob_failure_falls_back_to_algorithm_only(self, compare):
        compare.side_effect = [_evidence(.8), _evidence(1), _evidence(.8)]
        value, _ = confirm_knob(_spec(), 4096, "knomial",
                                Knob("K", "", "auto", ("4",)), "4")
        self.assertIsNone(value)


class TestSweepCell(unittest.TestCase):
    def test_sizeless_collective_is_rejected(self):
        # barrier reports count/size N/A, so the decision, the emitted range,
        # and its probes would be keyed on a size the workload never has.
        with self.assertRaisesRegex(ValueError, "no message-size dimension"):
            sweep_cell(_spec(collective="barrier"))
        with self.assertRaisesRegex(ValueError, "no message-size dimension"):
            sweep_cell(_spec(collective="barrier", proof_mode=True))

    @patch("ucc_tune_sweep.knobs_for",
           return_value=(Knob("K", "", "auto", ("4",)),))
    @patch("ucc_tune_sweep.confirm_knob")
    @patch("ucc_tune_sweep._paired_compare", return_value=_evidence(.8))
    @patch("ucc_tune_sweep.measure", return_value=_run(8))
    def test_knob_gates_join_family_and_budget(self, _measure, _paired,
                                               confirm, _knobs):
        # The three gates as production measures them: the knob-effect gate
        # carries the A0/A1 labels, the algorithm and joint gates are D/A.
        confirm.return_value = ("4", (_evidence(.8),
                                      _evidence(.8, arms=("A0", "A1")),
                                      _evidence(.8)))
        result = sweep_cell(_spec(proof_mode=True, max_confirmation_points=4))
        decision = result.size_decisions[0]
        self.assertEqual(decision.knob_overrides, {"K": "4"})
        self.assertEqual(len(decision.knob_hypotheses), 3)
        self.assertEqual(result.proof_budget.used_points, 4)
        audit = decision.knob_hypotheses[0].to_dict()
        self.assertEqual(set(("raw_p_value", "adjusted_threshold", "decision",
                              "reason")) - set(audit), set())

    @patch("ucc_tune_sweep.confirm_knob", return_value=(None, ()))
    @patch("ucc_tune_sweep._paired_compare")
    @patch("ucc_tune_sweep.measure")
    def test_screening_nomination_needs_fresh_pair_win(self, measure, paired, _knob):
        measure.side_effect = [_run(8), _run(10)]
        paired.return_value = _evidence(.8)
        result = sweep_cell(_spec(proof_mode=True))
        self.assertEqual(result.size_decisions[0].source, "fresh-paired-confirmation")
        self.assertTrue(result.size_decisions[0].should_override)
        self.assertEqual((result.tune_ranges[0].start_bytes,
                          result.tune_ranges[0].end_bytes), (4096, 4096))

    @patch("ucc_tune_sweep._paired_compare")
    @patch("ucc_tune_sweep.measure")
    def test_tie_emits_nothing(self, measure, paired):
        measure.side_effect = [_run(8), _run(10)]
        paired.return_value = _evidence(1)
        result = sweep_cell(_spec(proof_mode=True))
        self.assertEqual(result.tune_ranges, [])

    @patch("ucc_tune_sweep._paired_compare")
    @patch("ucc_tune_sweep.measure")
    def test_partial_advertised_sweep_cannot_emit(self, measure, paired):
        measure.side_effect = [_run(8), RuntimeError("failed"), _run(10)]
        result = sweep_cell(_spec(proof_mode=True,
                                  alg_list=[AlgInfo(0, "knomial", ""),
                                            AlgInfo(1, "ring", "")]))
        paired.assert_not_called()
        self.assertEqual(result.tune_ranges, [])
        self.assertIn("partial algorithm sweep", result.warnings[0])

    @patch("ucc_tune_sweep.confirm_knob", return_value=(None, ()))
    @patch("ucc_tune_sweep._paired_compare", return_value=_evidence(.8))
    @patch("ucc_tune_sweep.measure", return_value=_run(8))
    def test_aligned_duplicate_is_deduplicated(self, _measure, _paired, _knob):
        result = sweep_cell(_spec(msg_sizes_bytes=[4097, 4099]))
        self.assertEqual(len(result.size_decisions), 1)
        self.assertEqual(result.size_decisions[0].actual_size_bytes, 4096)
        self.assertTrue(any("deduplicated" in warning for warning in result.warnings))

    @patch("ucc_tune_sweep.confirm_knob")
    @patch("ucc_tune_sweep._paired_compare", return_value=_evidence(.8))
    @patch("ucc_tune_sweep.measure", return_value=_run(8))
    def test_skip_knobs_omits_knob_attribution(self, _measure, _paired, confirm):
        result = sweep_cell(_spec(proof_mode=True, skip_knobs=True))
        confirm.assert_not_called()
        self.assertTrue(result.size_decisions[0].should_override)
        self.assertEqual(result.size_decisions[0].knob_overrides, {})


class TestScreeningPath(unittest.TestCase):
    @patch("ucc_tune_sweep._paired_compare")
    @patch("ucc_tune_sweep.measure")
    def test_margin_win_emits_without_paired_confirmation(self, measure, paired):
        measure.side_effect = [_run(8), _run(10)]
        result = sweep_cell(_spec())  # proof_mode defaults to False
        paired.assert_not_called()
        self.assertEqual(result.size_decisions[0].source, "screening-margin")
        self.assertTrue(result.size_decisions[0].should_override)
        self.assertIsNone(result.proof_budget)
        self.assertEqual((result.tune_ranges[0].start_bytes,
                          result.tune_ranges[0].end_bytes), (4096, 4096))

    @patch("ucc_tune_sweep._paired_compare")
    @patch("ucc_tune_sweep.measure")
    def test_within_margin_emits_nothing(self, measure, paired):
        measure.side_effect = [_run(10), _run(10.2)]
        result = sweep_cell(_spec())
        paired.assert_not_called()
        self.assertEqual(result.tune_ranges, [])


class TestReadbackRecording(unittest.TestCase):
    @staticmethod
    def _run_rb(us, alg=None, comp=None, cv=0.01):
        sample = SingleRunSample(1024, 4096, us, us, us)
        sample.selected_alg = alg
        sample.selected_component = comp
        return RunResult(MagicMock(), [sample], us, 0, cv, 1, 0, 0, cv > .1,
                         selected_component=comp, selected_alg=alg)

    @patch("ucc_tune_sweep.measure")
    def test_default_readback_recorded_on_decision(self, measure):
        measure.side_effect = [
            self._run_rb(8, alg="knomial", comp="tl/ucp"),   # forced knomial
            self._run_rb(10, alg="knomial", comp="tl/ucp"),  # default arm
        ]
        result = sweep_cell(_spec())
        decision = result.size_decisions[0]
        self.assertEqual(decision.default_selected_alg, "knomial")
        self.assertEqual(decision.default_selected_component, "tl/ucp")
        self.assertEqual(decision.winner_cv, 0.01)
        self.assertEqual(decision.default_cv, 0.01)

    @patch("ucc_tune_sweep.measure")
    def test_readback_mismatch_warns(self, measure):
        measure.side_effect = [
            self._run_rb(8, alg="ring", comp="tl/ucp"),      # forced knomial, wrong readback
            self._run_rb(10, alg="knomial", comp="tl/ucp"),  # default arm
        ]
        result = sweep_cell(_spec())
        self.assertTrue(any("readback mismatch" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
