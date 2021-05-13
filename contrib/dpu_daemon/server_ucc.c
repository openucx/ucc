/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "server_ucc.h"
#include <assert.h>

// typedef struct ucc_test_oob_allgather_req {
//     ucc_ep_range_t range;
//     void *sbuf;
//     void *rbuf;
//     void *oob_coll_ctx;
//     int my_rank;
//     size_t msglen;
//     int iter;
//     MPI_Request reqs[2];
// } ucc_test_oob_allgather_req_t;

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *oob_coll_ctx, void **req)
{
    MPI_Comm    comm = (MPI_Comm)oob_coll_ctx;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)request;
    return UCC_OK;
}

ucc_status_t ucc_mpi_create_team_nb(dpu_ucc_comm_t *comm)
{
    ucc_status_t status = UCC_OK;
    /* Create UCC TEAM for comm world */
    ucc_team_params_t team_params = {
        .mask   = UCC_TEAM_PARAM_FIELD_EP |
                  UCC_TEAM_PARAM_FIELD_EP_RANGE |
                  UCC_TEAM_PARAM_FIELD_OOB,
        .ep     = comm->g->rank,
        .ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG,
        .oob   = {
            .allgather      = oob_allgather,
            .req_test       = oob_allgather_test,
            .req_free       = oob_allgather_free,
            .coll_info      = (void*)MPI_COMM_WORLD,
            .participants   = comm->g->size
        }
    };

    status = ucc_team_create_post(&comm->ctx, 1, &team_params, &comm->team);

    return status;
}

ucc_status_t ucc_mpi_create_team(dpu_ucc_comm_t *comm) {
    ucc_status_t status;
    
    status = ucc_mpi_create_team_nb(comm);
    while (UCC_INPROGRESS == (status = ucc_team_create_test(comm->team))) {
    };
    
    return status;
}

int dpu_ucc_init(int argc, char **argv, dpu_ucc_global_t *g)
{
    ucc_status_t status;
    char *var;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g->size);

    UCCCHECK_GOTO(ucc_lib_config_read("DPU_DAEMON", NULL, &g->lib_config),
                    exit_err, status);

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE |
                UCC_LIB_PARAM_FIELD_COLL_TYPES,
        .thread_mode = UCC_THREAD_SINGLE,
                /* TODO: support more collectives */
        .coll_types  = UCC_COLL_TYPE_ALLREDUCE,
    };

    UCCCHECK_GOTO(ucc_init(&lib_params, g->lib_config, &g->lib),
                    free_lib_config, status);

free_lib_config:
    ucc_lib_config_release(g->lib_config);
exit_err:
    return status;
}

int dpu_ucc_alloc_team(dpu_ucc_global_t *g, dpu_ucc_comm_t *comm)
{
    ucc_status_t status = UCC_OK;

    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params = {
        .mask   = UCC_CONTEXT_PARAM_FIELD_TYPE |
                  UCC_CONTEXT_PARAM_FIELD_OOB,
        .type   = UCC_CONTEXT_SHARED,
        .oob = {
            .allgather    = oob_allgather,
            .req_test     = oob_allgather_test,
            .req_free     = oob_allgather_free,
            .coll_info = (void*)MPI_COMM_WORLD,
            .participants = g->size
        },
    };
    ucc_context_config_h ctx_config;
    UCCCHECK_GOTO(ucc_context_config_read(g->lib, NULL, &ctx_config), free_ctx_config, status);
    UCCCHECK_GOTO(ucc_context_create(g->lib, &ctx_params, ctx_config, &comm->ctx), free_ctx, status);

    comm->g = g;
    UCCCHECK_GOTO(ucc_mpi_create_team(comm), free_ctx, status);

    return status;
free_ctx:
    ucc_context_destroy(comm->ctx);
free_ctx_config:
    ucc_context_config_release(ctx_config);

    return status;
}

int dpu_ucc_free_team(dpu_ucc_global_t *g, dpu_ucc_comm_t *team)
{
    ucc_team_destroy(team->team);
    ucc_context_destroy(team->ctx);
}

void dpu_ucc_finalize(dpu_ucc_global_t *g) {
    ucc_finalize(g->lib);
    MPI_Finalize();
}

void dpu_ucc_progress(dpu_ucc_comm_t *comm)
{
    ucc_context_progress(comm->ctx);
}
