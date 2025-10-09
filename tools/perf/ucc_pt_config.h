/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_CONFIG_H
#define UCC_PT_CONFIG_H

#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <getopt.h>
#include <ucc/api/ucc.h>
#include "utils/ucc_log.h"

#define UCC_PT_DEFAULT_N_BUFS 0

enum ucc_pt_bootstrap_type_t {
    UCC_PT_BOOTSTRAP_MPI,
    UCC_PT_BOOTSTRAP_UCX
};

struct ucc_pt_bootstrap_config {
    ucc_pt_bootstrap_type_t bootstrap;
};

struct ucc_pt_comm_config {
    ucc_memory_type_t mt;
};

typedef enum {
    UCC_PT_OP_TYPE_ALLGATHER       = UCC_COLL_TYPE_ALLGATHER,
    UCC_PT_OP_TYPE_ALLGATHERV      = UCC_COLL_TYPE_ALLGATHERV,
    UCC_PT_OP_TYPE_ALLREDUCE       = UCC_COLL_TYPE_ALLREDUCE,
    UCC_PT_OP_TYPE_ALLTOALL        = UCC_COLL_TYPE_ALLTOALL,
    UCC_PT_OP_TYPE_ALLTOALLV       = UCC_COLL_TYPE_ALLTOALLV,
    UCC_PT_OP_TYPE_BARRIER         = UCC_COLL_TYPE_BARRIER,
    UCC_PT_OP_TYPE_BCAST           = UCC_COLL_TYPE_BCAST,
    UCC_PT_OP_TYPE_FANIN           = UCC_COLL_TYPE_FANIN,
    UCC_PT_OP_TYPE_FANOUT          = UCC_COLL_TYPE_FANOUT,
    UCC_PT_OP_TYPE_GATHER          = UCC_COLL_TYPE_GATHER,
    UCC_PT_OP_TYPE_GATHERV         = UCC_COLL_TYPE_GATHERV,
    UCC_PT_OP_TYPE_REDUCE          = UCC_COLL_TYPE_REDUCE,
    UCC_PT_OP_TYPE_REDUCE_SCATTER  = UCC_COLL_TYPE_REDUCE_SCATTER,
    UCC_PT_OP_TYPE_REDUCE_SCATTERV = UCC_COLL_TYPE_REDUCE_SCATTERV,
    UCC_PT_OP_TYPE_SCATTER         = UCC_COLL_TYPE_SCATTER,
    UCC_PT_OP_TYPE_SCATTERV        = UCC_COLL_TYPE_SCATTERV,
    UCC_PT_OP_TYPE_MEMCPY          = UCC_COLL_TYPE_LAST + 1,
    UCC_PT_OP_TYPE_REDUCEDT,
    UCC_PT_OP_TYPE_REDUCEDT_STRIDED,
    UCC_PT_OP_TYPE_LAST
} ucc_pt_op_type_t;

typedef enum {
    UCC_PT_MAP_TYPE_NONE,
    UCC_PT_MAP_TYPE_LOCAL,
    UCC_PT_MAP_TYPE_GLOBAL,
    UCC_PT_MAP_TYPE_LAST
} ucc_pt_map_type_t;

typedef enum {
    UCC_PT_GEN_TYPE_EXP,
    UCC_PT_GEN_TYPE_FILE
} ucc_pt_gen_type_t;

static inline const char* ucc_pt_op_type_str(ucc_pt_op_type_t op)
{
    if ((uint64_t)op < (uint64_t)UCC_COLL_TYPE_LAST) {
        return ucc_coll_type_str((ucc_coll_type_t)op);
    }
    switch(op) {
    case UCC_PT_OP_TYPE_MEMCPY:
        return "Memcpy";
    case UCC_PT_OP_TYPE_REDUCEDT:
        return "Reduce DT";
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
        return "Reduce DT strided";
    default:
        break;
    }
    return NULL;
}

struct ucc_pt_gen_config {
    ucc_pt_gen_type_t type;
    size_t exp_min;
    size_t exp_max;
    std::string file_name;
    size_t nrep;  // Number of repetitions for file-based generation
};

struct ucc_pt_benchmark_config {
    ucc_pt_op_type_t   op_type;
    size_t             min_count;
    size_t             max_count;
    ucc_datatype_t     dt;
    ucc_memory_type_t  mt;
    ucc_reduction_op_t op;
    ucc_pt_map_type_t  map_type;
    bool               inplace;
    bool               persistent;
    bool               triggered;
    size_t             large_thresh;
    int                n_iter_small;
    int                n_warmup_small;
    int                n_iter_large;
    int                n_warmup_large;
    int                n_bufs;
    bool               full_print;
    int                root;
    int                root_shift;
    int                mult_factor;
    ucc_pt_gen_config  gen;
};

struct ucc_pt_config {
    ucc_pt_bootstrap_config bootstrap;
    ucc_pt_comm_config      comm;
    ucc_pt_benchmark_config bench;

    ucc_pt_config();
    ucc_status_t process_args(int argc, char *argv[]);
    void print_help();
};

#endif
