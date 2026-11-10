#!/usr/bin/env python3
"""Unit tests for ucc_tune_space (no ucc_info binary required)."""

import unittest

from ucc_tune_space import (
    AlgInfo,
    bytes_to_count,
    collective_count_plan,
    competition_env,
    count_plan_from_count,
    dtype_size,
    is_team_independent,
    is_unkeyable_collective,
    knobs_for,
    msg_size_grid,
    parse_ucc_info_algs,
    score_map_key_quantum,
    tune_env_var,
)

# ---------------------------------------------------------------------------
# Sample ucc_info -A output — reconstructed from the known print format:
#   section:  "%s/%s algorithms:\n"
#   coll:     "  %s\n"
#   alg:      "    %u : %16s : %s\n"
# ---------------------------------------------------------------------------

# Collective names are CAPITALISED here on purpose: ucc_info.c:62 prints them
# via ucc_coll_type_str() (src/utils/ucc_log.h:26). This block is copied
# byte-for-byte from a real `ucc_info -A` on hpcac-internal (UCC 1.9.0,
# 2026-07-28) so the fixture cannot re-encode the code's own assumptions.
_SAMPLE_OUTPUT = """\
tl/ucp algorithms:
  Allreduce
    0 :          knomial : recursive knomial with arbitrary radix (optimized for latency)
    1 :      sra_knomial : recursive knomial scatter-reduce followed by knomial allgather (optimized for BW)
    2 :              dbt : double binary tree
    3 :   sliding_window : sliding window allreduce (optimized for running on DPU)
    4 :             ring : ring-based allreduce using reduce-scatter + allgather
  Bcast
    0 :          knomial : recursive knomial bcast
    1 :      sag_knomial : scatter-allgather knomial bcast
  Reduce_scatter
    0 :          ring : O(N) ring reduce-scatter

tl/cuda algorithms:
  Allgather
    0 :             auto : choose allgather algorithm based on CUDA topology
    1 :             ring : multiring allgather algorithm
    2 :           linear : linear allgather algorithm
  Allreduce
    0 :             nvls : NVLINK SHARP allreduce

cl/hier algorithms:
  Allreduce
    0 :            2step : 2-step hierarchical allreduce

"""


class TestParseUccInfoAlgs(unittest.TestCase):
    def setUp(self):
        self.algs = parse_ucc_info_algs(_SAMPLE_OUTPUT)

    def test_components_present(self):
        self.assertIn("tl/ucp", self.algs)
        self.assertIn("tl/cuda", self.algs)
        self.assertIn("cl/hier", self.algs)

    def test_collectives_under_component(self):
        self.assertIn("allreduce", self.algs["tl/ucp"])
        self.assertIn("bcast", self.algs["tl/ucp"])
        self.assertIn("allgather", self.algs["tl/cuda"])
        self.assertIn("allreduce", self.algs["cl/hier"])

    def test_algorithm_count(self):
        self.assertEqual(len(self.algs["tl/ucp"]["allreduce"]), 5)
        self.assertEqual(len(self.algs["tl/ucp"]["bcast"]), 2)
        self.assertEqual(len(self.algs["tl/cuda"]["allgather"]), 3)

    def test_algorithm_ids_and_names(self):
        ar = self.algs["tl/ucp"]["allreduce"]
        self.assertEqual(ar[0], AlgInfo(id=0, name="knomial", desc="recursive knomial with arbitrary radix (optimized for latency)"))
        self.assertEqual(ar[1].name, "sra_knomial")
        self.assertEqual(ar[1].id, 1)
        self.assertEqual(ar[4].name, "ring")
        self.assertEqual(ar[4].id, 4)

    def test_cuda_nvls_alg(self):
        nvls = self.algs["tl/cuda"]["allreduce"]
        self.assertEqual(len(nvls), 1)
        self.assertEqual(nvls[0].name, "nvls")

    def test_hier_alg(self):
        h = self.algs["cl/hier"]["allreduce"]
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0].name, "2step")

    def test_collective_keys_are_lowercased(self):
        # Regression: ucc_info -A emits "Allreduce"/"Reduce_scatter", every
        # consumer looks up the lowercase perftest/TUNE spelling. Reproduced on
        # hpcac-internal 2026-07-28: without this, auto-discovery aborts with
        # "No components found" and --component skips every cell silently.
        for comp, colls in self.algs.items():
            for coll in colls:
                self.assertEqual(coll, coll.lower(),
                                 f"{comp}: collective key {coll!r} is not lowercase")
        self.assertNotIn("Allreduce", self.algs["tl/ucp"])
        self.assertIn("reduce_scatter", self.algs["tl/ucp"])

    def test_empty_output(self):
        self.assertEqual(parse_ucc_info_algs(""), {})

    def test_output_with_no_algs(self):
        out = "some unrelated output\nno sections here\n"
        self.assertEqual(parse_ucc_info_algs(out), {})

    def test_component_with_no_collectives(self):
        # A section header with nothing under it should produce an empty dict.
        out = "tl/self algorithms:\n\ntl/ucp algorithms:\n  barrier\n     0 :  knomial : barrier\n"
        algs = parse_ucc_info_algs(out)
        self.assertIn("tl/self", algs)
        self.assertEqual(algs["tl/self"], {})
        self.assertIn("tl/ucp", algs)
        self.assertEqual(len(algs["tl/ucp"]["barrier"]), 1)


class TestTuneEnvVar(unittest.TestCase):
    def test_tl_ucp(self):
        self.assertEqual(tune_env_var("tl/ucp"), "UCC_TL_UCP_TUNE")

    def test_tl_cuda(self):
        self.assertEqual(tune_env_var("tl/cuda"), "UCC_TL_CUDA_TUNE")

    def test_tl_nccl(self):
        self.assertEqual(tune_env_var("tl/nccl"), "UCC_TL_NCCL_TUNE")

    def test_cl_hier(self):
        self.assertEqual(tune_env_var("cl/hier"), "UCC_CL_HIER_TUNE")

    def test_cl_basic(self):
        self.assertEqual(tune_env_var("cl/basic"), "UCC_CL_BASIC_TUNE")


class TestCompetitionEnv(unittest.TestCase):
    def test_tl_ucp_restricts_tl_and_cl(self):
        env = competition_env("tl/ucp")
        self.assertEqual(env.get("UCC_TLS"), "ucp")
        self.assertEqual(env.get("UCC_CLS"), "basic")

    def test_tl_cuda_restricts_to_cuda(self):
        env = competition_env("tl/cuda")
        self.assertEqual(env.get("UCC_TLS"), "cuda")
        self.assertEqual(env.get("UCC_CLS"), "basic")

    def test_cl_hier_sets_cls_only(self):
        env = competition_env("cl/hier")
        self.assertEqual(env.get("UCC_CLS"), "hier")
        self.assertNotIn("UCC_TLS", env)

    def test_tl_env_does_not_set_tls_for_cl(self):
        # CL components should not restrict TLS since they may need any TL.
        env = competition_env("cl/basic")
        self.assertNotIn("UCC_TLS", env)


class TestKnobsFor(unittest.TestCase):
    def test_uint_ranged_metadata(self):
        knob = knobs_for("tl/ucp", "allreduce", "knomial")[0]
        self.assertEqual(knob.config_type, "UINT_RANGED")
        self.assertEqual(knob.scopes, ("message", "memory"))
        self.assertTrue(knob.range_scoped)

    def test_scalar_metadata(self):
        knob = knobs_for("tl/ucp", "alltoall", "pairwise")[0]
        self.assertEqual(knob.config_type, "SCALAR")
        self.assertFalse(knob.range_scoped)

    def test_allreduce_knomial_has_radix_knob(self):
        ks = knobs_for("tl/ucp", "allreduce", "knomial")
        self.assertEqual(len(ks), 1)
        self.assertEqual(ks[0].env_var, "UCC_TL_UCP_ALLREDUCE_KN_RADIX")
        self.assertIn("2", ks[0].candidates)
        self.assertIn("4", ks[0].candidates)
        self.assertIn("8", ks[0].candidates)

    def test_sra_knomial_has_radix_and_pipeline(self):
        ks = knobs_for("tl/ucp", "allreduce", "sra_knomial")
        env_vars = [k.env_var for k in ks]
        self.assertIn("UCC_TL_UCP_ALLREDUCE_SRA_KN_RADIX", env_vars)
        self.assertIn("UCC_TL_UCP_ALLREDUCE_SRA_KN_PIPELINE", env_vars)

    def test_pipeline_candidates_are_keyed_format(self):
        ks = knobs_for("tl/ucp", "allreduce", "sra_knomial")
        pipeline_knob = next(k for k in ks if "PIPELINE" in k.env_var)
        for cand in pipeline_knob.candidates:
            self.assertIn("thresh=", cand)
            self.assertIn("fragsize=", cand)
            self.assertIn("nfrags=", cand)
            self.assertIn("pdepth=", cand)

    def test_cuda_allgather_ring_knobs(self):
        ks = knobs_for("tl/cuda", "allgather", "ring")
        env_vars = [k.env_var for k in ks]
        self.assertIn("UCC_TL_CUDA_ALLGATHER_RING_MAX_RINGS", env_vars)
        self.assertIn("UCC_TL_CUDA_ALLGATHER_RING_NUM_CHUNKS", env_vars)

    def test_nvls_knobs(self):
        ks = knobs_for("tl/cuda", "allreduce", "nvls")
        env_vars = [k.env_var for k in ks]
        self.assertIn("UCC_TL_CUDA_NVLS_SM_COUNT", env_vars)
        self.assertIn("UCC_TL_CUDA_NVLS_THREADS", env_vars)

    def test_unknown_alg_returns_empty(self):
        self.assertEqual(knobs_for("tl/ucp", "allreduce", "nonexistent"), [])

    def test_unknown_component_returns_empty(self):
        self.assertEqual(knobs_for("tl/foobar", "allreduce", "knomial"), [])

    def test_all_candidates_are_strings(self):
        for (comp, coll, alg), knob_list in {
            ("tl/ucp", "allreduce", "sra_knomial"): None,
            ("tl/ucp", "allgather", "batched"): None,
            ("tl/cuda", "reduce_scatter", "ring"): None,
        }.items():
            for k in knobs_for(comp, coll, alg):
                for v in k.candidates:
                    self.assertIsInstance(v, str, f"{k.env_var} candidate {v!r} is not a string")


class TestMsgSizeGrid(unittest.TestCase):
    def test_basic_x2(self):
        grid = msg_size_grid(8, 64, factor=2)
        self.assertEqual(grid, [8, 16, 32, 64])

    def test_basic_x4(self):
        # 8 → 32 → 128 → 512 > 256 so loop ends; 256 appended as max.
        grid = msg_size_grid(8, 256, factor=4)
        self.assertEqual(grid, [8, 32, 128, 256])

    def test_basic_x4_correct(self):
        grid = msg_size_grid(8, 256, factor=4)
        self.assertEqual(grid[0], 8)
        self.assertEqual(grid[-1], 256)
        self.assertIn(32, grid)
        self.assertIn(128, grid)

    def test_exact_max_not_duplicated(self):
        grid = msg_size_grid(8, 32, factor=2)
        self.assertEqual(grid.count(32), 1)

    def test_single_point(self):
        grid = msg_size_grid(1024, 1024, factor=2)
        self.assertEqual(grid, [1024])

    def test_factor_below_2_raises(self):
        with self.assertRaises(ValueError):
            msg_size_grid(8, 1024, factor=1)

    def test_zero_min_raises_instead_of_hanging(self):
        # `s *= factor` keeps 0 at 0: without the guard the loop never ends.
        with self.assertRaises(ValueError):
            msg_size_grid(0, 1024, factor=2)

    def test_negative_min_raises_instead_of_hanging(self):
        # Negative steps grow away from a positive max, staying below it.
        with self.assertRaises(ValueError):
            msg_size_grid(-8, 1024, factor=2)

    def test_unordered_bounds_raise(self):
        with self.assertRaises(ValueError):
            msg_size_grid(1024, 8, factor=2)

    def test_default_grid_starts_at_8(self):
        grid = msg_size_grid()
        self.assertEqual(grid[0], 8)

    def test_default_grid_ends_at_1gib(self):
        grid = msg_size_grid()
        self.assertEqual(grid[-1], 1 << 30)

    def test_grid_is_monotonically_increasing(self):
        grid = msg_size_grid(8, 1 << 20, factor=2)
        for a, b in zip(grid, grid[1:]):
            self.assertLess(a, b)


class TestDtypeSize(unittest.TestCase):
    def test_float32(self):
        self.assertEqual(dtype_size("float32"), 4)

    def test_float64(self):
        self.assertEqual(dtype_size("float64"), 8)

    def test_bfloat16(self):
        self.assertEqual(dtype_size("bfloat16"), 2)

    def test_int8(self):
        self.assertEqual(dtype_size("int8"), 1)

    def test_float32_complex(self):
        self.assertEqual(dtype_size("float32_complex"), 8)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            dtype_size("fp8")


class TestBytesToCount(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(bytes_to_count(4096, "float32"), 1024)

    def test_rounds_down(self):
        self.assertEqual(bytes_to_count(4097, "float32"), 1024)

    def test_minimum_is_one(self):
        self.assertEqual(bytes_to_count(1, "float32"), 1)

    def test_float64(self):
        self.assertEqual(bytes_to_count(8192, "float64"), 1024)


class TestComplexDtypeWidths(unittest.TestCase):
    """ucc_dt.c:10-28 is the authority; only the *name* float128 is remapped.

    tools/perf/ucc_pt_config.cc maps "float128" to UCC_DT_FLOAT64, but
    "float128_complex" to the real 128-bit complex type, which ucc_dt_size()
    reports as 32.  Listing 16 doubled every count and labelled measured ranges
    with half the bytes perftest actually exercised.
    """

    def test_float128_complex_is_thirty_two_bytes(self):
        self.assertEqual(dtype_size("float128_complex"), 32)

    def test_float128_name_stays_at_the_float64_width(self):
        self.assertEqual(dtype_size("float128"), 8)

    def test_count_uses_the_true_complex_width(self):
        self.assertEqual(bytes_to_count(4096, "float128_complex"), 128)


class TestCollectiveCountPlan(unittest.TestCase):
    """Score-map keys follow ucc_coll_args_msgsize(), not count * dtype_size.

    Ground truth: src/utils/ucc_coll_utils.c:383 together with the counts that
    tools/perf/ucc_pt_coll_*.cc init_args() and the exponential generator
    install, and the Count column provenance in ucc_pt_benchmark.cc:171
    (print_time receives generator->get_src_count()).
    """

    def test_allgather_key_is_team_scaled_while_print_is_not(self):
        plan = collective_count_plan("allgather", "float32", 64, 8)
        self.assertEqual(plan.count, 2)                    # -b, not 64 // 4
        self.assertEqual(plan.score_map_bytes, 64)         # the key ranges match
        self.assertEqual(plan.expected_print_count, 2)
        self.assertEqual(plan.expected_print_size_bytes, 8)  # prints half the key

    def test_reduce_scatter_prints_a_team_scaled_source_count(self):
        plan = collective_count_plan("reduce_scatter", "float32", 64, 8)
        self.assertEqual(plan.count, 16)
        self.assertEqual(plan.score_map_bytes, 64)
        self.assertEqual(plan.expected_print_count, 128)
        self.assertEqual(plan.expected_print_size_bytes, 512)

    def test_alltoall_scales_both_key_and_print(self):
        plan = collective_count_plan("alltoall", "float32", 64, 8)
        self.assertEqual((plan.count, plan.score_map_bytes,
                          plan.expected_print_count,
                          plan.expected_print_size_bytes), (2, 64, 16, 64))

    def test_allreduce_stays_unscaled(self):
        plan = collective_count_plan("allreduce", "float32", 4096, 8)
        self.assertEqual((plan.count, plan.score_map_bytes,
                          plan.expected_print_size_bytes), (1024, 4096, 4096))

    def test_variable_collectives_key_on_total_dst_counts(self):
        for coll in ("allgatherv", "reduce_scatterv"):
            with self.subTest(coll=coll):
                plan = collective_count_plan(coll, "float32", 64, 8)
                self.assertEqual(plan.count, 2)
                self.assertEqual(plan.score_map_bytes, 64)

    def test_rooted_collectives_are_rank_consistent_under_perftest(self):
        # ucc_coll_args_msgsize multiplies the non-root side by the team size,
        # which cancels perftest's own per-rank scaling, so one key serves all
        # ranks and byte ranges remain expressible.
        for coll in ("gather", "scatter", "reduce", "bcast"):
            with self.subTest(coll=coll):
                plan = collective_count_plan(coll, "float32", 100, 8)
                self.assertEqual(plan.score_map_bytes % plan.key_quantum, 0)
                self.assertLessEqual(plan.score_map_bytes, 100)
                self.assertGreater(plan.score_map_bytes, 100 - plan.key_quantum)

    def test_key_is_floored_to_its_own_quantum(self):
        plan = collective_count_plan("allgather", "float32", 100, 8)
        self.assertEqual(plan.key_quantum, 32)
        self.assertEqual(plan.score_map_bytes, 96)

    def test_sizeless_collective_has_no_key(self):
        plan = collective_count_plan("barrier", "float32", 4096, 8)
        self.assertEqual((plan.score_map_bytes, plan.expected_print_count), (0, 0))

    def test_unkeyable_collectives_are_refused(self):
        # UCC_MSG_SIZE_ASYMMETRIC means the local arguments do not determine a
        # size at all, so a byte range could neither select nor validate work.
        for coll in ("alltoallv", "gatherv", "scatterv"):
            with self.subTest(coll=coll):
                self.assertTrue(is_unkeyable_collective(coll))
                with self.assertRaisesRegex(ValueError, "UCC_MSG_SIZE_ASYMMETRIC"):
                    collective_count_plan(coll, "float32", 4096, 8)

    def test_unknown_collective_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no score-map count model"):
            collective_count_plan("allgatherx", "float32", 64, 8)

    def test_team_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            collective_count_plan("allgather", "float32", 64, 0)

    def test_count_driven_view_agrees_with_size_driven_view(self):
        plan = count_plan_from_count("alltoall", "float32", 2, 8)
        self.assertEqual(plan.score_map_bytes, 64)
        self.assertEqual(plan.expected_print_size_bytes, 64)
        self.assertEqual(plan.expected_print_count, 16)

    def test_public_quantum_helper_matches_the_plan(self):
        self.assertEqual(score_map_key_quantum("alltoall", "float32", 8), 32)
        self.assertEqual(score_map_key_quantum("allreduce", "float32", 8), 4)

    def test_team_independence_classification(self):
        self.assertTrue(is_team_independent("allreduce"))
        self.assertTrue(is_team_independent("barrier"))
        self.assertFalse(is_team_independent("allgather"))
        self.assertFalse(is_team_independent("reduce_scatter"))
        self.assertFalse(is_team_independent("alltoallv"))


if __name__ == "__main__":
    unittest.main()
