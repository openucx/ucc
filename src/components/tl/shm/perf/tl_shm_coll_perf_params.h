/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm.h"
#include "../tl_shm_coll.h"

void ucc_tl_shm_perf_params_intel_broadwell_28_bcast(
    ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_broadwell_14_bcast(
    ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_broadwell_8_bcast(ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_skylake_40_bcast(ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_amd_rome_128_bcast(ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_broadwell_28_reduce(
    ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_broadwell_14_reduce(
    ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_broadwell_8_reduce(
    ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_intel_skylake_40_reduce(ucc_coll_task_t *coll_task);

void ucc_tl_shm_perf_params_amd_rome_128_reduce(ucc_coll_task_t *coll_task);

static inline void
ucc_tl_shm_perf_params_generic_bcast(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);

    task->progress_alg   = TASK_LIB(task)->cfg.bcast_alg;
    task->base_tree_only = TASK_LIB(task)->cfg.base_tree_only;
    task->base_radix     = TASK_LIB(task)->cfg.bcast_base_radix;
    task->top_radix      = TASK_LIB(task)->cfg.bcast_top_radix;
}

static inline void
ucc_tl_shm_perf_params_generic_reduce(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);

    task->base_tree_only = TASK_LIB(task)->cfg.base_tree_only;
    task->base_radix     = TASK_LIB(task)->cfg.reduce_base_radix;
    task->top_radix      = TASK_LIB(task)->cfg.reduce_top_radix;
}

static inline void
ucc_tl_shm_create_perf_func_list(ucc_tl_shm_team_t *      team,
                                 ucc_tl_shm_perf_keys_t * perf_funcs_keys,
                                 ucc_tl_shm_perf_funcs_t *perf_funcs_list)
{
    size_t size = 0;

    ucc_tl_shm_perf_keys_t intel_broadwell_28 = {
        .cpu_vendor  = UCC_CPU_VENDOR_INTEL,
        .cpu_model   = UCC_CPU_MODEL_INTEL_BROADWELL,
        .team_size   = 28,
        .layout      = SEG_LAYOUT_CONTIG,
        .bcast_func  = ucc_tl_shm_perf_params_intel_broadwell_28_bcast,
        .reduce_func = ucc_tl_shm_perf_params_intel_broadwell_28_reduce};
    perf_funcs_keys[size] = intel_broadwell_28;
    size++;

    ucc_tl_shm_perf_keys_t intel_broadwell_14 = {
        .cpu_vendor  = UCC_CPU_VENDOR_INTEL,
        .cpu_model   = UCC_CPU_MODEL_INTEL_BROADWELL,
        .team_size   = 14,
        .layout      = UCC_TL_SHM_TEAM_LIB(team)->cfg.layout,
        .bcast_func  = ucc_tl_shm_perf_params_intel_broadwell_14_bcast,
        .reduce_func = ucc_tl_shm_perf_params_intel_broadwell_14_reduce};
    perf_funcs_keys[size] = intel_broadwell_14;
    size++;

    ucc_tl_shm_perf_keys_t intel_broadwell_8 = {
        .cpu_vendor  = UCC_CPU_VENDOR_INTEL,
        .cpu_model   = UCC_CPU_MODEL_INTEL_BROADWELL,
        .team_size   = 8,
        .layout      = UCC_TL_SHM_TEAM_LIB(team)->cfg.layout,
        .bcast_func  = ucc_tl_shm_perf_params_intel_broadwell_8_bcast,
        .reduce_func = ucc_tl_shm_perf_params_intel_broadwell_8_reduce};
    perf_funcs_keys[size] = intel_broadwell_8;
    size++;

    ucc_tl_shm_perf_keys_t intel_skylake_40 = {
        .cpu_vendor  = UCC_CPU_VENDOR_INTEL,
        .cpu_model   = UCC_CPU_MODEL_INTEL_SKYLAKE,
        .team_size   = 40,
        .layout      = SEG_LAYOUT_SOCKET,
        .bcast_func  = ucc_tl_shm_perf_params_intel_skylake_40_bcast,
        .reduce_func = ucc_tl_shm_perf_params_intel_skylake_40_reduce};
    perf_funcs_keys[size] = intel_skylake_40;
    size++;

    ucc_tl_shm_perf_keys_t amd_rome_128 = {
        .cpu_vendor  = UCC_CPU_VENDOR_AMD,
        .cpu_model   = UCC_CPU_MODEL_AMD_ROME,
        .team_size   = 128,
        .layout      = SEG_LAYOUT_SOCKET,
        .bcast_func  = ucc_tl_shm_perf_params_amd_rome_128_bcast,
        .reduce_func = ucc_tl_shm_perf_params_amd_rome_128_reduce};
    perf_funcs_keys[size] = amd_rome_128;
    size++;

    perf_funcs_list->size = size;
    perf_funcs_list->keys = perf_funcs_keys;
}
