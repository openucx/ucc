/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include <assert.h>

ucc_team_h       ucc_world_team;
ucc_context_h    team_ctx;
static ucc_lib_h lib;

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)request;
    return UCC_OK;
}

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

int ucc_mpi_create_comm_nb(MPI_Comm comm, ucc_team_h *team)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Create UCC TEAM for comm world */
    ucc_team_params_t team_params = {.mask = UCC_TEAM_PARAM_FIELD_EP |
                                             UCC_TEAM_PARAM_FIELD_OOB,
                                     .oob = {.allgather    = oob_allgather,
                                             .req_test     = oob_allgather_test,
                                             .req_free     = oob_allgather_free,
                                             .coll_info    = (void *)comm,
                                             .participants = size},
                                     .ep  = rank};

    UCC_CHECK(ucc_team_create_post(&team_ctx, 1, &team_params, team));
    return 0;
}

int ucc_mpi_create_comm(MPI_Comm comm, ucc_team_h *team)
{
    ucc_mpi_create_comm_nb(comm, team);
    while (UCC_INPROGRESS == ucc_team_create_test(*team)) {
        ;
    };
    return 0;
}

int ucc_mpi_test_init(int argc, char **argv, uint64_t coll_types,
                      unsigned thread_mode)
{
    char *var;
    int   rank, size, provided;
    int   required = (thread_mode == UCC_THREAD_SINGLE) ? MPI_THREAD_SINGLE
                                                        : MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, required, &provided);
    assert(provided == required);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Init ucc library */
    var = getenv("UCC_TEST_CLS");

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        /* .coll_types = coll_types, */
    };

    ucc_lib_config_h lib_config;

    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params = {
        .mask     = UCC_CONTEXT_PARAM_FIELD_TYPE,
        .ctx_type = UCC_CONTEXT_EXCLUSIVE,
    };
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    if (var) {
        UCC_CHECK(ucc_lib_config_modify(lib_config, "CLS", var));
    }
    UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);

    ucc_context_config_h ctx_config;
    UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));
    UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &team_ctx));
    ucc_context_config_release(ctx_config);
    ucc_mpi_create_comm(MPI_COMM_WORLD, &ucc_world_team);
    return 0;
}

void ucc_mpi_test_finalize(void)
{
    UCC_CHECK(ucc_team_destroy(ucc_world_team));
    UCC_CHECK(ucc_context_destroy(team_ctx));
    UCC_CHECK(ucc_finalize(lib));
    MPI_Finalize();
}

void ucc_mpi_test_progress(void)
{
    /* ucc_context_progress(team_ctx); */
}
