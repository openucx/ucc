/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
#include "../alltoallv/alltoallv.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_alltoall_algs[UCC_CL_HIER_ALLTOALL_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLTOALL_ALG_NODE_SPLIT] =
            {.id   = UCC_CL_HIER_ALLTOALL_ALG_NODE_SPLIT,
             .name = "node_split",
             .desc = "splitting alltoall into two concurrent a2av calls"
                     " withing the node and outside of it"},
        [UCC_CL_HIER_ALLTOALL_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_alltoall_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    int                     i, count;
    ucc_rank_t              team_size;
    ucc_mc_buffer_header_t *h;

    if (UCC_IS_INPLACE(coll_args->args)) {
        cl_debug(team->context->lib, "inplace alltoall is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!SBGP_ENABLED(cl_team, NODE) || !SBGP_ENABLED(cl_team, FULL)) {
        cl_debug(team->context->lib, "alltoall requires NODE and FULL sbgps");
        return UCC_ERR_NOT_SUPPORTED;
    }

    memcpy(&args, coll_args, sizeof(args));
    args.args.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    if (!(args.args.mask & UCC_COLL_ARGS_FIELD_FLAGS)) {
        args.args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags = 0;
    }
    args.args.flags |= UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                       UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    team_size = UCC_CL_TEAM_SIZE(cl_team);

    status =
        ucc_mc_alloc(&h, sizeof(int) * team_size * 2, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for full counts",
                 sizeof(int) * team_size * 2);
        return status;
    }

    args.args.src.info_v.buffer   = coll_args->args.src.info.buffer;
    args.args.dst.info_v.buffer   = coll_args->args.dst.info.buffer;
    args.args.src.info_v.datatype = coll_args->args.src.info.datatype;
    args.args.dst.info_v.datatype = coll_args->args.dst.info.datatype;
    args.args.src.info_v.mem_type = coll_args->args.src.info.mem_type;
    args.args.dst.info_v.mem_type = coll_args->args.dst.info.mem_type;

    args.args.src.info_v.counts = h->addr;
    args.args.src.info_v.displacements =
        PTR_OFFSET(h->addr, sizeof(int) * team_size);

    args.args.dst.info_v.counts        = args.args.src.info_v.counts;
    args.args.dst.info_v.displacements = args.args.src.info_v.displacements;

    count = (int)coll_args->args.src.info.count / team_size;
    ((int *)args.args.src.info_v.counts)[0]        = count;
    ((int *)args.args.src.info_v.displacements)[0] = 0;

    for (i = 1; i < team_size; i++) {
        ((int *)args.args.src.info_v.counts)[i] = count;
        ((int *)args.args.src.info_v.displacements)[i] =
            ((int *)args.args.src.info_v.displacements)[i - 1] + count;
    }
    status = ucc_cl_hier_alltoallv_init(&args, team, task);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to init split node a2av task");
    }

    ucc_mc_free(h);
    return status;
}
