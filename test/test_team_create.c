/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <api/ucc.h>
#include <stdio.h>
#include <mpi.h>

int test_allgather(void *src_buf, void *recv_buf, size_t size,
                   void *allgather_info,  void **request) {
    printf("running test oob allgather\n");
    return UCC_OK;
}

ucc_status_t test_req_test(void *request) {
    printf("running test oob req\n");
    return UCC_OK;
}

ucc_status_t test_req_free(void *request) {
    printf("running test req free\n");
    return UCC_OK;
}

int main(int argc, char **argv) {
    ucc_lib_config_h config;
    ucc_context_config_h ctx_config;
    ucc_lib_h lib;
    ucc_context_h ucc_ctx;
    ucc_status_t status;
    int rank, size;
    ucc_team_h team;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ucc_context_params_t ctx_params = {
        .mask = 0,
    };

    ucc_lib_params_t params = {
        .mask = 0,
    };

    ucc_team_oob_coll_t oob = {
        .allgather = test_allgather,
        .req_test  = test_req_test,
        .req_free  = test_req_free,
        .participants = size,
        .coll_info = (void*)MPI_COMM_WORLD,
    };

    ucc_team_params_t team_params = {
        .mask = UCC_TEAM_PARAM_FIELD_OOB,
        .oob = oob,
    };

    status = ucc_lib_config_read("TEST", NULL, &config);
    if (UCC_OK != status) {
        goto error;
    }

    status = ucc_lib_init(&params, config, &lib);
    if (UCC_OK != status) {
        goto error;
    }

    ucc_lib_config_release(config);

    status = ucc_context_config_read(lib, NULL, &ctx_config);
    if (UCC_OK != status) {
        goto error;
    }

    status = ucc_context_create(lib, &ctx_params, ctx_config, &ucc_ctx);
    if (UCC_OK != status) {
        goto error;
    }
    ucc_context_config_release(ctx_config);

    status = ucc_team_create_post(&ucc_ctx, 1, &team_params, &team);
    if (UCC_OK != status) {
        goto error;
    }

    while (UCC_OK != ucc_team_create_test(team)) {;}

    ucc_team_destroy(team);

    ucc_context_destroy(ucc_ctx);
    ucc_lib_cleanup(lib);
    MPI_Finalize();
    return 0;
error:
    return -1;
}
