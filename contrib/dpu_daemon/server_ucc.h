/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TEST_MPI_H
#define TEST_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

#include <ucc/api/ucc.h>

#define STR(x) #x

#define UCC_CHECK(_call) if (UCC_OK != (_call)) {              \
        fprintf(stderr, "*** UCC TEST FAIL: %s\n", STR(_call)); \
        MPI_Abort(MPI_COMM_WORLD, -1);                           \
    }

#define UCCCHECK_GOTO(_call, _label, _status)                                  \
    do {                                                                       \
        _status = (_call);                                                     \
        if (UCC_OK != _status) {                                               \
            fprintf(stderr, "UCC DPU DAEMON error: %s\n", STR(_call));         \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

typedef struct {
    ucc_team_h          ucc_world_team;
    ucc_lib_h           lib;
    ucc_lib_config_h    lib_config;
    int rank;
    int size;
} dpu_ucc_global_t;

typedef struct {
    dpu_ucc_global_t *g;
    ucc_context_h ctx;
    ucc_team_h team;
} dpu_ucc_comm_t;

int dpu_ucc_init(int argc, char **argv, dpu_ucc_global_t *g);
int dpu_ucc_alloc_team(dpu_ucc_global_t *g, dpu_ucc_comm_t *team);
int dpu_ucc_free_team(dpu_ucc_global_t *g, dpu_ucc_comm_t *ctx);
void dpu_ucc_finalize(dpu_ucc_global_t *g);
void dpu_ucc_progress(dpu_ucc_comm_t *team);

#endif
