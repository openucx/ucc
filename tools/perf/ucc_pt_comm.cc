#include <iostream>
#include <cstring>
#include "ucc_pt_comm.h"
#include "ucc_pt_bootstrap_mpi.h"
#include "ucc_perftest.h"
#include "ucc_pt_cuda.h"
#include "ucc_pt_rocm.h"
extern "C" {
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
}

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
    int dev_count = 0;

    if (ucc_pt_cudaGetDeviceCount(&dev_count) == 0 && dev_count != 0) {
        ucc_pt_cudaSetDevice(bootstrap->get_local_rank() % dev_count);
        return;
    }

    if (ucc_pt_rocmGetDeviceCount(&dev_count) == 0 && dev_count != 0) {
        ucc_pt_rocmSetDevice(bootstrap->get_local_rank() % dev_count);
    }

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

ucc_ee_h ucc_pt_comm::get_ee()
{
    ucc_ee_params_t ee_params;
    ucc_status_t status;

    if (!ee) {
        if (cfg.mt == UCC_MEMORY_TYPE_CUDA) {
            if (ucc_pt_cudaStreamCreateWithFlags((cudaStream_t*)&stream,
                                                 cudaStreamNonBlocking)) {
                throw std::runtime_error("failed to create CUDA stream");
            }
            ee_params.ee_type         = UCC_EE_CUDA_STREAM;
            ee_params.ee_context_size = sizeof(cudaStream_t);
            ee_params.ee_context      = stream;
            status = ucc_ee_create(team, &ee_params, &ee);
            if (status != UCC_OK) {
                std::cerr << "failed to create UCC EE: "
                          << ucc_status_string(status);
                ucc_pt_cudaStreamDestroy((cudaStream_t)stream);
                throw std::runtime_error(ucc_status_string(status));
            }
        } else {
            std::cerr << "execution engine is not supported for given memory type"
                      << std::endl;
            throw std::runtime_error("not supported");
        }
    }

    return ee;
}

ucc_ee_executor_t* ucc_pt_comm::get_executor()
{
    ucc_ee_executor_params_t executor_params;
    ucc_status_t             status;

    if (!executor) {
        executor_params.mask = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
        if (cfg.mt ==  UCC_MEMORY_TYPE_HOST) {
            executor_params.ee_type = UCC_EE_CPU_THREAD;
        } else if (cfg.mt == UCC_MEMORY_TYPE_CUDA) {
            executor_params.ee_type = UCC_EE_CUDA_STREAM;
        } else if (cfg.mt == UCC_MEMORY_TYPE_ROCM) {
            executor_params.ee_type = UCC_EE_ROCM_STREAM;
        } else {
            std::cerr << "executor is not supported for given memory type"
                      << std::endl;
            throw std::runtime_error("not supported");
        }
        status = ucc_ee_executor_init(&executor_params, &executor);
        if (status != UCC_OK) {
            throw std::runtime_error("failed to init executor");
        }
    }
    return executor;
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
    std::string cfg_mod;

    ee       = nullptr;
    executor = nullptr;
    stream   = nullptr;

    if (cfg.mt != UCC_MEMORY_TYPE_HOST) {
        set_gpu_device();
    }
    UCCCHECK_GOTO(ucc_lib_config_read("PERFTEST", nullptr, &lib_config),
                  exit_err, st);
    std::memset(&lib_params, 0, sizeof(ucc_lib_params_t));
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    UCCCHECK_GOTO(ucc_init(&lib_params, lib_config, &lib), free_lib_config, st);

    if (UCC_OK != ucc_mc_available(cfg.mt)) {
        std::cerr << "selected memory type " << ucc_mem_type_str(cfg.mt) <<
            " is not available" << std::endl;
        return UCC_ERR_INVALID_PARAM;
    }

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

    if (ee) {
        status = ucc_ee_destroy(ee);
        if (status != UCC_OK) {
            std::cerr << "ucc ee destroy error: " << ucc_status_string(status);
        }

        if (cfg.mt == UCC_MEMORY_TYPE_CUDA) {
            ucc_pt_cudaStreamDestroy((cudaStream_t)stream);
        } else {
            std::cerr << "execution engine is not supported for given memory type"
                      << std::endl;
            throw std::runtime_error("not supported");
        }
    }

    if (executor) {
        status = ucc_ee_executor_finalize(executor);
        if (status != UCC_OK) {
            std::cerr << "ucc executor finalize error: "
                      << ucc_status_string(status);
        }
    }

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

ucc_status_t ucc_pt_comm::allreduce(double* in, double* out, size_t size,
                                    ucc_reduction_op_t op)
{
    ucc_coll_args_t args;
    ucc_coll_req_h req;

    args.mask                 = 0;
    args.coll_type            = UCC_COLL_TYPE_ALLREDUCE;
    args.op                   = op;
    args.src.info.buffer      = in;
    args.src.info.count       = size;
    args.src.info.datatype    = UCC_DT_FLOAT64;
    args.src.info.mem_type    = UCC_MEMORY_TYPE_HOST;
    args.dst.info.buffer      = out;
    args.dst.info.count       = size;
    args.dst.info.datatype    = UCC_DT_FLOAT64;
    args.dst.info.mem_type    = UCC_MEMORY_TYPE_HOST;
    ucc_collective_init(&args, &req, team);
    ucc_collective_post(req);
    do {
        ucc_context_progress(context);
    } while (ucc_collective_test(req) == UCC_INPROGRESS);
    ucc_collective_finalize(req);
    return UCC_OK;
}
