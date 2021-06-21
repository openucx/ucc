#include <iostream>
#include <cstring>
#include "ucc_pt_comm.h"
#include "ucc_pt_bootstrap_mpi.h"
#include "ucc_perftest.h"

ucc_pt_comm::ucc_pt_comm(ucc_pt_comm_config config)
{
    cfg = config;
    bootstrap = new ucc_pt_bootstrap_mpi();
}

ucc_pt_comm::~ucc_pt_comm()
{
    delete bootstrap;
}

void ucc_pt_comm::set_gpu_device()
{
#ifdef HAVE_CUDA
    cudaError_t st;
    int dev_count;
    CUDA_CHECK_GOTO(cudaGetDeviceCount(&dev_count), exit_cuda, st);
    if (dev_count == 0) {
        return;
    }
    CUDA_CHECK_GOTO(cudaSetDevice(bootstrap->get_local_rank() % dev_count),
                    exit_cuda, st);
exit_cuda:
#endif
    return;
}

int ucc_pt_comm::get_rank()
{
    return bootstrap->get_rank();
}

int ucc_pt_comm::get_size()
{
    return bootstrap->get_size();
}

ucc_team_h ucc_pt_comm::get_team(int team_id)
{
    return teams.at(team_id);
}

ucc_context_h ucc_pt_comm::get_context()
{
    return context;
}

ucc_status_t ucc_pt_comm::init(int n_threads)
{
    ucc_lib_config_h lib_config;
    ucc_context_config_h ctx_config;
    ucc_lib_params_t lib_params;
    ucc_context_params_t ctx_params;
    ucc_team_params_t team_params;
    ucc_status_t st;
    std::string cfg_mod;
    n_teams = n_threads;
    int i;

    if (cfg.mt != UCC_MEMORY_TYPE_HOST) {
        set_gpu_device();
    }
    UCCCHECK_GOTO(ucc_lib_config_read("PERFTEST", nullptr, &lib_config),
                  exit_err, st);
    std::memset(&lib_params, 0, sizeof(ucc_lib_params_t));
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode =
        (n_threads > 1) ? UCC_THREAD_MULTIPLE : UCC_THREAD_SINGLE;
    UCCCHECK_GOTO(ucc_init(&lib_params, lib_config, &lib), free_lib_config, st);
    UCCCHECK_GOTO(ucc_context_config_read(lib, NULL, &ctx_config),
                  free_lib, st);
    cfg_mod = std::to_string(bootstrap->get_size());
    UCCCHECK_GOTO(ucc_context_config_modify(ctx_config, NULL,
                  "ESTIMATED_NUM_EPS", cfg_mod.c_str()), free_ctx_config, st);
    cfg_mod = std::to_string(bootstrap->get_ppn());
    UCCCHECK_GOTO(ucc_context_config_modify(ctx_config, NULL,
                  "ESTIMATED_NUM_PPN", cfg_mod.c_str()), free_ctx_config, st);
    std::memset(&ctx_params, 0, sizeof(ucc_context_params_t));
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE |
                      UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.type = UCC_CONTEXT_SHARED;
    ctx_params.oob  = bootstrap->get_context_oob();
    UCCCHECK_GOTO(ucc_context_create(lib, &ctx_params, ctx_config, &context),
                  free_ctx_config, st);
    teams.resize(n_teams);
    team_params.mask     = UCC_TEAM_PARAM_FIELD_EP |
                           UCC_TEAM_PARAM_FIELD_EP_RANGE |
                           UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob      = bootstrap->get_team_oob();
    team_params.ep       = bootstrap->get_rank();
    team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    for (i = 0; i < n_teams; i++) {
        UCCCHECK_GOTO(
            ucc_team_create_post(&context, 1, &team_params, &teams.at(i)),
            free_teams, st);
        do {
            st = ucc_team_create_test(teams.at(i));
        } while (st == UCC_INPROGRESS);
        UCCCHECK_GOTO(st, free_teams, st);
    }
    ucc_context_config_release(ctx_config);
    ucc_lib_config_release(lib_config);
    return UCC_OK;
free_teams:
    for (i = i - 1; i >= 0; i--) {
        do {
            st = ucc_team_destroy(teams.at(i));
        } while (st == UCC_INPROGRESS);
        if (st != UCC_OK) {
            std::cerr << "ucc team destroy error: " << ucc_status_string(st);
        }
    }
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
    int          i;

    for (i = 0; i < n_teams; i++) {
        do {
            status = ucc_team_destroy(teams.at(i));
        } while (status == UCC_INPROGRESS);
        if (status != UCC_OK) {
            std::cerr << "ucc team destroy error: "
                      << ucc_status_string(status);
        }
    }
    ucc_context_destroy(context);
    ucc_finalize(lib);
    return UCC_OK;
}

ucc_status_t ucc_pt_comm::barrier(int team_id)
{
    ucc_coll_args_t args;
    ucc_coll_req_h req;

    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_BARRIER;
    ucc_collective_init(&args, &req, teams.at(team_id));
    ucc_collective_post(req);
    do {
        ucc_context_progress(context);
    } while (ucc_collective_test(req) == UCC_INPROGRESS);
    ucc_collective_finalize(req);
    return UCC_OK;
}

ucc_status_t ucc_pt_comm::allreduce(float* in, float* out, size_t size,
                                    ucc_reduction_op_t op, int team_id)
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
    ucc_collective_init(&args, &req, teams.at(team_id));
    ucc_collective_post(req);
    do {
        ucc_context_progress(context);
    } while (ucc_collective_test(req) == UCC_INPROGRESS);
    ucc_collective_finalize(req);
    return UCC_OK;
}
