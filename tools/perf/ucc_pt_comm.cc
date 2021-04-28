#include <iostream>
#include <cstring>
#include "ucc_pt_comm.h"
#include "ucc_pt_bootstrap_mpi.h"
#include "ucc_perftest.h"

ucc_pt_comm::ucc_pt_comm()
{
    bootstrap = new ucc_pt_bootstrap_mpi();
}

ucc_pt_comm::~ucc_pt_comm()
{
    delete bootstrap;
}

int ucc_pt_comm::get_rank()
{
    return bootstrap->get_rank();
}

int ucc_pt_comm::get_size()
{
    return bootstrap->get_size();
}

ucc_team_h ucc_pt_comm::get_team()
{
    return team;
}

ucc_context_h ucc_pt_comm::get_context()
{
    return context;
}

ucc_status_t ucc_pt_comm::init()
{
    ucc_lib_config_h lib_config;
    ucc_context_config_h ctx_config;
    ucc_lib_params_t lib_params;
    ucc_context_params_t ctx_params;
    ucc_team_params_t team_params;
    ucc_status_t st;

    UCCCHECK_GOTO(ucc_lib_config_read("PERFTEST", nullptr, &lib_config),
                  exit_err, st);
    std::memset(&lib_params, 0, sizeof(ucc_lib_params_t));
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    UCCCHECK_GOTO(ucc_init(&lib_params, lib_config, &lib), free_lib_config, st);
    UCCCHECK_GOTO(ucc_context_config_read(lib, NULL, &ctx_config),
                  free_lib, st);
    std::memset(&ctx_params, 0, sizeof(ucc_context_params_t));
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE |
                          UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.type = UCC_CONTEXT_SHARED;
    ctx_params.oob  = bootstrap->get_context_oob();
    UCCCHECK_GOTO(ucc_context_create(lib, &ctx_params, ctx_config, &context),
                  free_ctx_config, st);
    team_params.mask     = UCC_TEAM_PARAM_FIELD_EP |
                           UCC_TEAM_PARAM_FIELD_EP_RANGE |
                           UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob      = bootstrap->get_team_oob();
    team_params.ep       = bootstrap->get_rank();
    team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    UCCCHECK_GOTO(ucc_team_create_post(&context, 1, &team_params, &team),
                  free_ctx, st);
    do {
        st = ucc_team_create_test(team);
    } while(st == UCC_INPROGRESS);
    UCCCHECK_GOTO(st, free_ctx, st);
    ucc_context_config_release(ctx_config);
    ucc_lib_config_release(lib_config);
    return UCC_OK;
free_ctx:
    ucc_context_destroy(context);
free_ctx_config:
    ucc_context_config_release(ctx_config);
free_lib:
    ucc_finalize(lib);
free_lib_config:
    ucc_lib_config_release(lib_config);
exit_err:
    return st;
}

ucc_status_t ucc_pt_comm::finalize()
{
    ucc_status_t status;

    do {
        status = ucc_team_destroy(team);
    } while (status == UCC_INPROGRESS);
    if (status != UCC_OK) {
        std::cerr << "ucc team destroy error: " << ucc_status_string(status);
    }
    ucc_context_destroy(context);
    ucc_finalize(lib);
    return UCC_OK;
}

ucc_status_t ucc_pt_comm::barrier()
{
    ucc_coll_args_t args;
    ucc_coll_req_h req;

    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_BARRIER;
    ucc_collective_init(&args, &req, team);
    ucc_collective_post(req);
    do {
        ucc_context_progress(context);
    } while (ucc_collective_test(req) == UCC_INPROGRESS);
    ucc_collective_finalize(req);
    return UCC_OK;
}

ucc_status_t ucc_pt_comm::allreduce(float* in, float* out, size_t size,
                                    ucc_reduction_op_t op)
{
    ucc_coll_args_t args;
    ucc_coll_req_h req;

    args.mask                 = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
    args.coll_type            = UCC_COLL_TYPE_ALLREDUCE;
    args.reduce.predefined_op = op;
    args.src.info.buffer      = in;
    args.src.info.count       = size;
    args.src.info.datatype    = UCC_DT_FLOAT32;
    args.src.info.mem_type    = UCC_MEMORY_TYPE_HOST;
    args.dst.info.buffer      = out;
    args.dst.info.count       = size;
    args.dst.info.datatype    = UCC_DT_FLOAT32;
    args.dst.info.mem_type    = UCC_MEMORY_TYPE_HOST;
    ucc_collective_init(&args, &req, team);
    ucc_collective_post(req);
    do {
        ucc_context_progress(context);
    } while (ucc_collective_test(req) == UCC_INPROGRESS);
    ucc_collective_finalize(req);
    return UCC_OK;
}
