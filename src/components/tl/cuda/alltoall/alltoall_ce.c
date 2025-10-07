/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "../alltoallv/alltoallv.h"
#include "alltoall.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

//NOLINTNEXTLINE(misc-unused-parameters): cnts and block parameters unused in alltoall variant
size_t ucc_tl_cuda_alltoall_get_size(const ucc_tl_cuda_task_t *task,
                                     size_t *cnts, ucc_rank_t block)
{
    (void)cnts;  /* Unused parameter */
    (void)block; /* Unused parameter */
    return ucc_dt_size(TASK_ARGS(task).dst.info.datatype) *
           (TASK_ARGS(task).dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task)));
}

size_t ucc_tl_cuda_alltoall_get_offset(const ucc_tl_cuda_task_t *task,
                                       size_t *displ, ucc_rank_t block)
{
    (void)displ; /* Unused parameter */
    return ucc_dt_size(TASK_ARGS(task).dst.info.datatype) *
           (TASK_ARGS(task).dst.info.count /
            UCC_TL_TEAM_SIZE(TASK_TEAM(task))) *
           block;
}

ucc_status_t ucc_tl_cuda_alltoall_ce_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib  = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_status_t        status;
    size_t              data_len;

    task->alltoallv_ce.get_size   = ucc_tl_cuda_alltoall_get_size;
    task->alltoallv_ce.get_offset = ucc_tl_cuda_alltoall_get_offset;
    task->alltoallv_ce.sdt        = args->src.info.datatype;
    task->alltoallv_ce.rdt        = args->dst.info.datatype;
    task->alltoallv_ce.sbuf       = args->src.info.buffer;
    task->alltoallv_ce.rbuf       = args->dst.info.buffer;
    task->alltoallv_ce.stage      = 0;
    /* NOT used for alltoall */
    task->alltoallv_ce.scnts  = 0;
    task->alltoallv_ce.rcnts  = 0;
    task->alltoallv_ce.sdispl = 0;
    task->alltoallv_ce.rdispl = 0;

    data_len = ucc_dt_size(args->src.info.datatype) * args->src.info.count;
    status   = ucc_tl_cuda_mem_info_get(
        args->src.info.buffer, data_len, &task->alltoallv_ce.mem_info_src);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    if (team->topo->proxy_needed) {
        status = ucc_tl_cuda_mem_info_get(
            args->dst.info.buffer, data_len, &task->alltoallv_ce.mem_info_dst);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    return ucc_tl_cuda_alltoallv_ce_setup_copy_engine(
        task, lib, "ucc_tl_cuda_alltoall_ce_init");
}
