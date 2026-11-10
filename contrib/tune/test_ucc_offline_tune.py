#!/usr/bin/env python3
"""Tests for scoped emission and exact-config validation."""

import dataclasses
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ucc_offline_tune import (
    ValidationPoint, _build_arg_parser, _build_findings, _collect_knob_overrides,
    _collect_tune_tokens, _fmt_duration, _results_to_json,
    _validation_probe_sizes, _validation_covers_results, compute_cost_model,
    emit_conf, print_cost_model, trim_failed_ranges, validate,
    validate_with_trimming, run_tuning, write_findings, _compute_cell_budget,
)
from ucc_tune_fingerprint import Fingerprint, _stable_hash
from ucc_tune_runner import PairedRunResult
from ucc_tune_space import AlgInfo
from ucc_tune_stats import ArmSample, Decision, classify_cell, classify_evidence
from ucc_tune_sweep import SizeDecision, SweepResult, SweepSpec, TuneRange

def fingerprint(revision="0123456789ab"):
    fp = Fingerprint("1.4", "1.17", "cpu", "gpu", "driver", "cuda",
                     "host", "2026-08-05T00:00:00Z", "",
                     ucc_revision=revision)
    return dataclasses.replace(fp, hash=_stable_hash(fp))



def tune_range(start=4096, end=4100, knobs=None, alg="knomial"):
    return TuneRange(start, end, alg, 0, knobs or {})


def win_decision(size=4096):
    return SizeDecision(
        size_bytes=size, should_override=True, winner_name="knomial",
        winner_id=0, winner_median_us=8.0, default_median_us=10.0,
        margin=0.2, knob_overrides={}, policy=Decision.WIN, evidence=None,
        actual_size_bytes=size, source="screening-margin")


def sweep_result(*, ranges=None, mem="host", team=8, all_teams=None,
                 sizes=None, collective="allreduce", component="tl/ucp",
                 datatype="float32", reduction_op="sum", decisions=None):
    spec = SweepSpec(
        component, collective, mem, team, sizes or [4096, 4100, 16384],
        [AlgInfo(0, "knomial", "")], all_team_sizes=all_teams or [team],
        datatype=datatype, reduction_op=reduction_op,
    )
    return SweepResult(spec, decisions if decisions is not None else [],
                       ranges or [], [])


def paired(ratio=.8):
    arm_samples = []
    for i in range(10):
        order = "AB" if i % 2 == 0 else "BA"
        arm_samples.extend((ArmSample(i, order, "D", 100),
                            ArmSample(i, order, "A", 100 * ratio)))
    evidence = classify_evidence(arm_samples)
    return PairedRunResult(tuple(arm_samples), 10, 10, 0, evidence)


class TestTuneTokens(unittest.TestCase):
    def test_exact_inclusive_token_and_team(self):
        tokens = _collect_tune_tokens([sweep_result(ranges=[tune_range()])])
        self.assertEqual(tokens["UCC_TL_UCP_TUNE"],
                         ["allreduce:4k-4100:host:[8-8]:inf:@knomial"])

    def test_zero_width_range_emits_no_token(self):
        tokens = _collect_tune_tokens(
            [sweep_result(ranges=[tune_range(4096, 4096)])])
        self.assertEqual(tokens, {})

    def test_unmeasured_team_sizes_are_not_covered(self):
        results = [sweep_result(team=8, all_teams=[8, 64], ranges=[tune_range()]),
                   sweep_result(team=64, all_teams=[8, 64], ranges=[tune_range()])]
        tokens = _collect_tune_tokens(results)["UCC_TL_UCP_TUNE"]
        self.assertIn("[8-8]", tokens[0])
        self.assertIn("[64-64]", tokens[1])
        self.assertNotIn("inf", tokens[0].split(":")[3])

    def test_unencodable_mixed_datatype_or_operation_is_rejected(self):
        for variant in (sweep_result(datatype="float64"),
                        sweep_result(reduction_op="max")):
            with self.assertRaisesRegex(ValueError, "cannot emit"):
                _collect_tune_tokens([
                    sweep_result(ranges=[tune_range()]),
                    variant,
                ])


class TestKnobEmission(unittest.TestCase):
    RADIX = "UCC_TL_UCP_ALLREDUCE_KN_RADIX"

    def test_uint_ranged_keeps_message_and_memory_scope(self):
        results = [sweep_result(ranges=[tune_range(knobs={self.RADIX: "2"})]),
                   sweep_result(mem="cuda", ranges=[tune_range(8192, 8196,
                                                               {self.RADIX: "4"})])]
        env, warnings = _collect_knob_overrides(results)
        self.assertEqual(warnings, [])
        self.assertEqual(env[self.RADIX], "4k-4100:host:2,8k-8196:cuda:4,auto")

    def test_ranged_knob_is_omitted_across_teams(self):
        results = [sweep_result(team=8, ranges=[tune_range(knobs={self.RADIX: "4"})]),
                   sweep_result(team=64, ranges=[tune_range(knobs={self.RADIX: "4"})])]
        env, warnings = _collect_knob_overrides(results)
        self.assertNotIn(self.RADIX, env)
        self.assertTrue(any("team-scoped" in warning for warning in warnings))

    def test_overlapping_conflict_is_omitted(self):
        result = sweep_result(ranges=[
            tune_range(4096, 8192, {self.RADIX: "2"}),
            tune_range(8192, 16380, {self.RADIX: "4"}),
        ])
        env, warnings = _collect_knob_overrides([result])
        self.assertNotIn(self.RADIX, env)
        self.assertTrue(any("conflict" in warning for warning in warnings))

    def test_scalar_partial_range_is_omitted(self):
        env, warnings = _collect_knob_overrides([
            sweep_result(ranges=[tune_range(knobs={"UNKNOWN_SCALAR": "4"})])])
        self.assertEqual(env, {})
        self.assertTrue(warnings)

    def test_scalar_full_single_cell_is_allowed(self):
        result = sweep_result(sizes=[4096, 4100],
                              ranges=[tune_range(knobs={"UNKNOWN_SCALAR": "4"})])
        env, _ = _collect_knob_overrides([result])
        self.assertEqual(env, {"UNKNOWN_SCALAR": "4"})

    def test_scalar_does_not_leak_into_an_unrelated_memory_cell(self):
        host = sweep_result(sizes=[4096, 4100],
                            ranges=[tune_range(knobs={"UNKNOWN_SCALAR": "4"})])
        cuda = sweep_result(mem="cuda", sizes=[4096, 4100], ranges=[])
        env, warnings = _collect_knob_overrides([host, cuda])
        self.assertNotIn("UNKNOWN_SCALAR", env)
        self.assertTrue(warnings)


class TestFingerprintRevision(unittest.TestCase):
    def test_parses_version_and_revision_from_build_banner(self):
        import ucc_tune_fingerprint
        banner = ("# UCC version=1.7.0 revision deadbeefcafe\n"
                   "# Configured with: --prefix=/opt/ucc\n")
        with patch("ucc_tune_fingerprint._run", return_value=banner):
            self.assertEqual(ucc_tune_fingerprint._ucc_build_info("ucc_info"),
                             ("1.7.0", "deadbeefcafe"))

    def test_collected_fingerprints_differ_by_revision(self):
        import ucc_tune_fingerprint

        def run_for(revision):
            def run(cmd, **_kwargs):
                if cmd[0] == "ucc_info":
                    return f"# UCC version=1.7.0 revision {revision}\n"
                return ""
            return run

        with patch("ucc_tune_fingerprint._run", side_effect=run_for("aaaa1111")):
            first = ucc_tune_fingerprint.collect()
        with patch("ucc_tune_fingerprint._run", side_effect=run_for("bbbb2222")):
            second = ucc_tune_fingerprint.collect()
        self.assertEqual(first.ucc_version, second.ucc_version)
        self.assertNotEqual(first.ucc_revision, second.ucc_revision)
        self.assertNotEqual(first.hash, second.hash)

    def test_proof_mode_bounded_analysis_after_max_pairs(self):
        # 20 pairs is the supported maximum; exhaustive sign-flip enumeration
        # would leave this cell computing for minutes after measurement ends.
        defaults = [ArmSample(p, "AB" if p % 2 else "BA", "D", 100.0)
                    for p in range(20)]
        candidates = [ArmSample(p, "AB" if p % 2 else "BA", "A", 80.0)
                      for p in range(20)]
        strong = defaults + candidates
        start = time.monotonic()
        decisions = classify_cell([strong, strong])
        self.assertLess(time.monotonic() - start, 30)
        self.assertTrue(all(e.decision is Decision.WIN for e in decisions))


class TestValidationProbes(unittest.TestCase):
    def test_complete_probe_set_and_inclusive_end(self):
        probes = _validation_probe_sizes(
            [tune_range(4096, 16380)],
            [1024, 4096, 4100, 16380, 16384, 16388],
            resolution_bytes=1024,
        )
        by_size = {(probe.size_bytes, probe.inside): probe for probe in probes}
        for key in ((4096, True), (16380, True), (4100, True),
                    (4092, False), (16384, False)):
            self.assertIn(key, by_size)
        reasons = {reason for probe in probes for reason in probe.reasons}
        self.assertTrue({"midpoint", "quartile-25", "quartile-75"} <= reasons)

    def test_alignment_deduplicates_probes(self):
        probes = _validation_probe_sizes([tune_range(4096, 4100)],
                                         [4096, 4097, 4099, 4100])
        keys = [(probe.size_bytes, probe.inside) for probe in probes]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(size % 4 == 0 for size, _ in keys))

    @patch("ucc_offline_tune.measure_paired")
    def test_exact_config_validation_inside_and_outside(self, measure):
        result = sweep_result(ranges=[tune_range()], sizes=[4092, 4096, 4100, 4104])
        measure.side_effect = lambda default, tuned, **kwargs: (
            paired(.8) if 4096 <= default.count * 4 <= 4100 else paired(1.0))
        points = validate([result], _collect_tune_tokens([result]), {}, n_reps=10)
        self.assertTrue(points)
        self.assertTrue(all(point.passed for point in points))
        self.assertTrue(any(not point.inside for point in points))

    @patch("ucc_offline_tune.measure_paired")
    def test_arms_are_the_deployed_config_and_unmodified_ucc(self, measure):
        """The artifact carries no competition-isolation vars, so validating an
        isolated arm would certify a configuration nobody deploys."""
        result = sweep_result(
            ranges=[tune_range(knobs={"UCC_TL_UCP_ALLREDUCE_KN_RADIX": "2"})],
            sizes=[4092, 4096, 4100, 4104])
        tokens = _collect_tune_tokens([result])
        knob_env, _ = _collect_knob_overrides([result])
        self.assertEqual(tokens["UCC_TL_UCP_TUNE"],
                         ["allreduce:4k-4100:host:[8-8]:inf:@knomial"])
        self.assertTrue(knob_env)
        emitted = {var: "#".join(values) for var, values in tokens.items()}
        emitted.update(knob_env)
        seen = []
        measure.side_effect = lambda default, tuned, **kwargs: (
            seen.append((default, tuned)) or paired())
        validate([result], tokens, knob_env, n_reps=10)
        self.assertTrue(seen)
        for default_spec, tuned_spec in seen:
            self.assertEqual(tuned_spec.extra_env, emitted)
            self.assertEqual(default_spec.extra_env, {})
            self.assertNotIn("UCC_TLS", tuned_spec.extra_env)
            self.assertNotIn("UCC_CLS", tuned_spec.extra_env)

    def test_trim_then_remove_cap(self):
        result = sweep_result(ranges=[tune_range(4096, 8192)], sizes=[4096, 8192])
        points = [ValidationPoint("allreduce", "host", 4096, 8, 10, .2, True,
                                  component="tl/ucp", requested_team_size=8,
                                  executed_team_size=8),
                  ValidationPoint("allreduce", "host", 8192, 11, 10, -.1, False,
                                  component="tl/ucp", requested_team_size=8,
                                  executed_team_size=8)]
        trim_failed_ranges([result], points, 1)
        self.assertEqual((result.tune_ranges[0].start_bytes,
                          result.tune_ranges[0].end_bytes), (4096, 4096))
        result.tune_ranges = [tune_range(4096, 8192)]
        trim_failed_ranges([result], points, 2)
        self.assertEqual(result.tune_ranges, [])
        with self.assertRaises(ValueError):
            trim_failed_ranges([result], points, 3)

    def test_trim_isolated_by_complete_cell_identity_in_both_rounds(self):
        variants = [
            sweep_result(component="tl/ucp", team=8,
                         ranges=[tune_range(4096, 8192)]),
            sweep_result(component="tl/nccl", team=8,
                         ranges=[tune_range(4096, 8192)]),
            sweep_result(component="tl/ucp", team=64,
                         ranges=[tune_range(4096, 8192)]),
        ]
        points = [
            ValidationPoint("allreduce", "host", size, 8, 10, .2, passed,
                            component=component, team_size=team,
                            requested_team_size=team, executed_team_size=team)
            for component, team, size, passed in (
                ("tl/ucp", 8, 4096, True), ("tl/ucp", 8, 8192, False),
                ("tl/nccl", 8, 4096, True), ("tl/nccl", 8, 8192, True),
                ("tl/ucp", 64, 4096, True), ("tl/ucp", 64, 8192, True),
            )
        ]
        trim_failed_ranges(variants, points, 1)
        self.assertEqual([(r.tune_ranges[0].start_bytes,
                           r.tune_ranges[0].end_bytes) for r in variants],
                         [(4096, 4096), (4096, 8192), (4096, 8192)])

        variants[0].tune_ranges = [tune_range(4096, 8192)]
        trim_failed_ranges(variants, points, 2)
        self.assertEqual(variants[0].tune_ranges, [])
        self.assertTrue(all(r.tune_ranges for r in variants[1:]))

    def test_trim_isolated_by_datatype_and_reduction_operation(self):
        variants = [
            sweep_result(datatype="float32", reduction_op="sum",
                         ranges=[tune_range(4096, 8192)]),
            sweep_result(datatype="float64", reduction_op="sum",
                         ranges=[tune_range(4096, 8192)]),
            sweep_result(datatype="float32", reduction_op="max",
                         ranges=[tune_range(4096, 8192)]),
        ]
        points = []
        for result in variants:
            spec = result.spec
            fail = spec.datatype == "float32" and spec.reduction_op == "sum"
            for size in (4096, 8192):
                points.append(ValidationPoint(
                    "allreduce", "host", size, 8, 10, .2, not fail,
                    component="tl/ucp", team_size=8, requested_team_size=8,
                    executed_team_size=8, datatype=spec.datatype,
                    reduction_op=spec.reduction_op))
        trim_failed_ranges(variants, points, 1)
        self.assertEqual(variants[0].tune_ranges, [])
        self.assertTrue(all(r.tune_ranges for r in variants[1:]))

    def test_coverage_ignores_opposing_outcome_from_other_cell(self):
        result = sweep_result(ranges=[tune_range(4096, 4096)], sizes=[4096])
        own = ValidationPoint(
            "allreduce", "host", 4096, 8, 10, .2, True,
            component="tl/ucp", requested_team_size=8, executed_team_size=8)
        foreign = ValidationPoint(
            "allreduce", "host", 4096, 11, 10, -.1, False,
            component="tl/nccl", requested_team_size=8, executed_team_size=8)
        self.assertTrue(_validation_covers_results([result], [own, foreign]))

    @patch("ucc_offline_tune.measure_paired", return_value=paired(.8))
    def test_outside_performance_difference_fails(self, _measure):
        result = sweep_result(ranges=[tune_range()], sizes=[4092, 4096, 4100, 4104])
        points = validate([result], _collect_tune_tokens([result]), {})
        self.assertTrue(any(not point.inside and not point.passed for point in points))

    @patch("ucc_offline_tune.validate")
    def test_validation_regeneration_is_bounded(self, validate_mock):
        result = sweep_result(ranges=[tune_range(4096, 8192)], sizes=[4096, 8192])
        validate_mock.side_effect = [
            [ValidationPoint("allreduce", "host", 4096, 8, 10, .2, True,
                             component="tl/ucp", requested_team_size=8,
                             executed_team_size=8),
             ValidationPoint("allreduce", "host", 8192, 11, 10, -.1, False,
                             component="tl/ucp", requested_team_size=8,
                             executed_team_size=8)],
            [ValidationPoint("allreduce", "host", 4096, 11, 10, -.1, False,
                             component="tl/ucp", requested_team_size=8,
                             executed_team_size=8)],
        ]
        validate_with_trimming([result])
        self.assertLessEqual(validate_mock.call_count, 3)
        self.assertEqual(result.tune_ranges, [])


class TestEmission(unittest.TestCase):
    def test_default_output_is_clearly_provisional(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = emit_conf(Path(directory),
                              [sweep_result(ranges=[tune_range()])], fingerprint())
            self.assertEqual(paths["conf"].name, "ucc_tuned_provisional.conf")
            self.assertIn("not-for-deployment", paths["conf"].read_text())
            payload = json.loads(paths["results"].read_text())
            self.assertEqual(payload["status"], "provisional-not-for-deployment")

    def test_accepted_requires_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                emit_conf(Path(directory), [], fingerprint(), accepted=True)
            result = sweep_result(ranges=[tune_range(4096, 4096)], sizes=[4096])
            paths = emit_conf(
                Path(directory), [result], fingerprint(), accepted=True,
                correctness={"default_complete": True, "tuned_complete": True,
                             "default_failures": [], "new_tuned_failures": []},
                validation_points=[ValidationPoint("allreduce", "host", 4096,
                                                   8, 10, .2, True,
                                                   component="tl/ucp", team_size=8,
                                                   requested_team_size=8,
                                                   executed_team_size=8)],
            )
            self.assertEqual(paths["conf"].name, "ucc_tuned.conf")
            payload = json.loads(paths["results"].read_text())
            self.assertEqual(payload["validation"][0]["cell_key"]["datatype"],
                             "float32")

    def test_json_is_deterministic_and_inclusive(self):
        result = sweep_result(ranges=[tune_range()])
        first = json.dumps(_results_to_json([result]), sort_keys=True)
        second = json.dumps(_results_to_json([result]), sort_keys=True)
        self.assertEqual(first, second)
        self.assertTrue(_results_to_json([result])[0]["tune_ranges"][0]["inclusive"])
        self.assertEqual(_results_to_json([result])[0]["cell_key"], {
            "component": "tl/ucp", "collective": "allreduce",
            "mem_type": "host", "team_size": 8,
            "datatype": "float32", "reduction_op": "sum",
        })


class TestCliSafety(unittest.TestCase):
    def test_routine_defaults(self):
        args = _build_arg_parser().parse_args([])
        self.assertEqual((args.min_speedup, args.min_pairs, args.max_pairs),
                         (.05, 10, 20))
        self.assertEqual((args.boundary_resolution_bytes,
                          args.max_boundary_probes,
                          args.max_confirmation_points,
                          args.per_cell_min_points), (1024, 4, 40, 10))
        self.assertEqual(args.team_sizes, "8")

    def test_proof_mode_flag_exists_without_unsafe_flag(self):
        parser = _build_arg_parser()
        self.assertTrue(parser.parse_args(["--proof-mode"]).proof_mode)
        self.assertNotIn("legacy", parser.format_help().lower())

    @patch("ucc_offline_tune.collect_fingerprint")
    def test_unsupported_launcher_fails_before_stage_zero(self, fingerprint):
        from ucc_offline_tune import main
        with self.assertRaises(SystemExit):
            main(["--launcher", "jsrun -n 8"])
        fingerprint.assert_not_called()

    @patch("ucc_offline_tune.write_findings")
    @patch("ucc_offline_tune.write_summary")
    @patch("ucc_offline_tune.emit_conf")
    @patch("ucc_offline_tune.run_tuning")
    @patch("ucc_offline_tune.run_ucc_info_algs")
    @patch("ucc_offline_tune.collect_fingerprint")
    def test_successful_run_returns_zero(self, collect, algs, run, emit,
                                         summary, findings):
        from ucc_offline_tune import main
        collect.return_value = fingerprint()
        algs.return_value = {"tl/ucp": {"allreduce": [AlgInfo(0, "knomial", "")]}}
        run.return_value = ([sweep_result(ranges=[tune_range()],
                                          decisions=[win_decision(4096)])], [])
        emit.return_value = {"conf": Path("/tmp/c"), "sh": Path("/tmp/s"),
                             "fingerprint": Path("/tmp/f"), "results": Path("/tmp/r")}
        summary.return_value = Path("/tmp/summary")
        findings.return_value = {"json": Path("/tmp/fj"), "md": Path("/tmp/fm")}
        self.assertEqual(
            main(["--component", "tl/ucp", "--collective", "allreduce",
                  "--no-validate"]),
            0,
        )

    @patch("ucc_offline_tune.write_findings")
    @patch("ucc_offline_tune.write_summary")
    @patch("ucc_offline_tune.emit_conf")
    @patch("ucc_offline_tune.run_tuning")
    @patch("ucc_offline_tune.run_ucc_info_algs")
    @patch("ucc_offline_tune.collect_fingerprint")
    def test_unmeasured_cells_return_one(self, collect, algs, run, emit,
                                         summary, findings):
        """Launcher/perftest failures must not look like a clean no-op run."""
        from ucc_offline_tune import main
        collect.return_value = fingerprint()
        algs.return_value = {"tl/ucp": {"allreduce": [AlgInfo(0, "knomial", "")]}}
        run.return_value = ([sweep_result()], [
            "tl/ucp/allreduce mem=host team_size=8: NO MEASUREMENTS"])
        emit.return_value = {"conf": Path("/tmp/c"), "sh": Path("/tmp/s"),
                             "fingerprint": Path("/tmp/f"), "results": Path("/tmp/r")}
        summary.return_value = Path("/tmp/summary")
        findings.return_value = {"json": Path("/tmp/fj"), "md": Path("/tmp/fm")}
        self.assertEqual(
            main(["--component", "tl/ucp", "--collective", "allreduce",
                  "--no-validate"]),
            1,
        )

    @patch("ucc_offline_tune.collect_fingerprint")
    def test_sizeless_collective_rejected_before_stage_zero(self, fingerprint):
        from ucc_offline_tune import main
        with self.assertRaises(SystemExit):
            main(["--component", "tl/ucp", "--collective", "barrier"])
        fingerprint.assert_not_called()

    @patch("ucc_offline_tune.collect_fingerprint")
    def test_nonpositive_min_bytes_rejected_before_stage_zero(self, fingerprint):
        from ucc_offline_tune import main
        with self.assertRaises(SystemExit):
            main(["--component", "tl/ucp", "--collective", "allreduce",
                  "--min-bytes", "0"])
        fingerprint.assert_not_called()

    @patch("ucc_offline_tune.collect_fingerprint")
    def test_inverted_byte_bounds_rejected_before_stage_zero(self, fingerprint):
        from ucc_offline_tune import main
        with self.assertRaises(SystemExit):
            main(["--component", "tl/ucp", "--collective", "allreduce",
                  "--min-bytes", "4096", "--max-bytes", "8"])
        fingerprint.assert_not_called()


class TestCellLaunchers(unittest.TestCase):
    @patch("ucc_offline_tune.sweep_cell")
    def test_two_team_sizes_get_distinct_bound_launchers(self, sweep):
        sweep.side_effect = lambda spec: SweepResult(spec, [], [], [])
        run_tuning([("tl/ucp", "allreduce")], ["host"], [8, 64], [4096],
                   alg_map={"tl/ucp": {"allreduce": [AlgInfo(0, "knomial", "")]}},
                   mpi_launcher=["mpirun", "-np", "{team_size}"])
        specs = [call.args[0] for call in sweep.call_args_list]
        self.assertEqual([spec.mpi_launcher for spec in specs],
                         [["mpirun", "-np", "8"], ["mpirun", "-np", "64"]])
        self.assertEqual([spec.executed_team_size for spec in specs], [8, 64])

    @patch("ucc_offline_tune.sweep_cell")
    def test_confirmation_seed_threaded_to_spec(self, sweep):
        sweep.side_effect = lambda spec: SweepResult(spec, [], [], [])
        run_tuning([("tl/ucp", "allreduce")], ["host"], [8], [4096],
                   alg_map={"tl/ucp": {"allreduce": [AlgInfo(0, "knomial", "")]}},
                   mpi_launcher=["mpirun", "-np", "{team_size}"],
                   confirmation_seed=42)
        self.assertEqual(sweep.call_args.args[0].confirmation_seed, 42)

    def test_serialized_requested_and_executed_team_size(self):
        payload = _results_to_json([sweep_result()])[0]
        self.assertEqual(payload["requested_team_size"], 8)
        self.assertEqual(payload["executed_team_size"], 8)

class TestConfirmationBudget(unittest.TestCase):
    """Regression: `done` was incremented before the per-cell budget was
    computed, so single-cell runs got max_points=0 and every size fell back
    to screening-only (no paired confirmation).  The budget must be computed
    from the count of cells already completed.
    """
    @patch("ucc_offline_tune.sweep_cell")
    def test_single_cell_gets_full_budget(self, sweep):
        sweep.side_effect = lambda spec: SweepResult(spec, [], [], [])
        run_tuning([("tl/ucp", "allreduce")], ["host"], [8], [4096],
                   alg_map={"tl/ucp": {"allreduce": [AlgInfo(0, "knomial", "")]}},
                   mpi_launcher=["mpirun", "-np", "{team_size}"])
        self.assertEqual(sweep.call_args.args[0].max_confirmation_points, 40)

    def test_four_cells_floor_then_divide_surplus(self):
        # Exact fit: the 40-point budget with a 10-point per-cell floor lands
        # 10/10/10/10 across four cells (no surplus to divide).
        budgets = []
        used = 0
        for done in range(4):
            budget = _compute_cell_budget(4, done, 40, used, 10)
            budgets.append(budget)
            used += budget
        self.assertEqual(budgets, [10, 10, 10, 10])

        # With surplus, the leftover is divided equally on top of the floor.
        budgets = []
        used = 0
        for done in range(4):
            budget = _compute_cell_budget(4, done, 48, used, 10)
            budgets.append(budget)
            used += budget
        self.assertEqual(budgets, [12, 12, 12, 12])


class TestCostModel(unittest.TestCase):
    ALG_MAP = {
        "tl/ucp": {
            "allreduce": [AlgInfo(0, "knomial", "a"), AlgInfo(1, "sra_knomial", "b"),
                          AlgInfo(2, "ring", "c"), AlgInfo(3, "dbt", "d"),
                          AlgInfo(4, "sliding_window", "e")],
            "bcast": [AlgInfo(0, "knomial", "a")],
        },
    }

    def test_screening_formula(self):
        # 2 collectives x 2 mem_types x 2 team_sizes = 8 cells.
        # allreduce: 5 algs -> 6 arms; bcast: 1 alg -> 2 arms.
        # sizes = [8, 16, 32] -> 3 sizes; n_reps = 7.
        model = compute_cost_model(
            [("tl/ucp", "allreduce"), ("tl/ucp", "bcast")],
            self.ALG_MAP, ["host", "cuda"], [8, 64], [8, 16, 32], 7,
            skip_asymmetric=False,
        )
        # allreduce cells: 4 cells x 3 sizes x 6 arms x 7 = 504
        # bcast cells:     4 cells x 3 sizes x 2 arms x 7 = 168
        self.assertEqual(model["total_invocations"], 504 + 168)
        self.assertEqual(model["cells"], 8)
        self.assertEqual(model["sizes"], 3)

    def test_asymmetric_collectives_skipped(self):
        model = compute_cost_model(
            [("tl/ucp", "allreduce"), ("tl/ucp", "bcast")],
            self.ALG_MAP, ["host"], [8], [8, 16], 3,
            skip_asymmetric=True,
        )
        # bcast is asymmetric (rooted) and skipped by default.
        self.assertEqual(model["cells"], 1)
        self.assertEqual(model["skipped_cells"], 1)
        # allreduce: 1 cell x 2 sizes x 6 arms x 3 = 36
        self.assertEqual(model["total_invocations"], 36)

    def test_fmt_duration(self):
        self.assertEqual(_fmt_duration(45), "45s")
        self.assertEqual(_fmt_duration(90), "1m 30s")
        self.assertEqual(_fmt_duration(3661), "1h 1m 1s")

    def test_print_cost_model_mentions_total(self):
        import contextlib
        import io
        model = compute_cost_model(
            [("tl/ucp", "allreduce")], self.ALG_MAP, ["host"], [8], [8, 16], 3,
            skip_asymmetric=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_cost_model(model, 2.0)
        self.assertIn("total invocations", buf.getvalue())
        self.assertIn("est. wall-clock", buf.getvalue())


class TestFindings(unittest.TestCase):
    def _decision(self, size, margin, alg="sra_knomial", default_alg="knomial",
                  cv=0.03):
        return SizeDecision(
            size_bytes=size, should_override=True, winner_name=alg, winner_id=1,
            winner_median_us=8.0, default_median_us=10.0, margin=margin,
            knob_overrides={}, policy=Decision.WIN, evidence=None,
            actual_size_bytes=size, source="screening-margin",
            default_selected_alg=default_alg,
            default_selected_component="tl/ucp",
            winner_cv=cv,
        )

    def _result(self, ranges):
        spec = SweepSpec(
            component="tl/ucp", collective="allreduce", mem_type="host",
            team_size=8, msg_sizes_bytes=[4096, 8192], alg_list=[],
        )
        return SweepResult(spec, [], ranges, [])

    def test_ranked_by_speedup(self):
        r1 = TuneRange(4096, 4096, "sra_knomial", 1, {},
                       (self._decision(4096, 0.12),))
        r2 = TuneRange(8192, 8192, "knomial", 0, {},
                       (self._decision(8192, 0.20),))
        leads = _build_findings([self._result([r1, r2])])
        self.assertEqual(len(leads), 2)
        self.assertEqual(leads[0]["winning_algorithm"], "knomial")
        self.assertAlmostEqual(leads[0]["speedup"], 0.20)
        self.assertEqual(leads[0]["default_algorithm"], "knomial")
        self.assertEqual(leads[1]["winning_algorithm"], "sra_knomial")
        self.assertEqual(leads[1]["rank"], 2)

    def test_empty_results(self):
        self.assertEqual(_build_findings([]), [])
        self.assertEqual(_build_findings([self._result([])]), [])

    def test_write_findings_files(self):
        r = TuneRange(4096, 8192, "sra_knomial", 1, {},
                      (self._decision(4096, 0.12), self._decision(8192, 0.15)))
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_findings(Path(tmp), [self._result([r])], fingerprint())
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["md"].exists())
            data = json.loads(paths["json"].read_text())
            self.assertEqual(data["leads"][0]["winning_algorithm"], "sra_knomial")
            self.assertEqual(data["status"], "provisional-leads-not-for-deployment")
            self.assertIn("human triage", paths["md"].read_text())


if __name__ == "__main__":
    unittest.main()
