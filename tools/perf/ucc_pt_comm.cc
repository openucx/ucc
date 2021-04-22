#include <iostream>
#include <cstring>
#include "ucc_pt_comm.h"
#include "ucc_pt_bootstrap_mpi.h"

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
    ucc_context_config_h context_config;
    ucc_lib_params_t lib_params;
    ucc_context_params_t context_params;
    ucc_team_params_t team_params;
    ucc_status_t st;
    st = ucc_lib_config_read("TORCH", nullptr, &lib_config);

    if (st != UCC_OK) {
        std::cerr << "failed to read UCC lib config: " << ucc_status_string(st);
        return st;
    }
    std::memset(&lib_params, 0, sizeof(ucc_lib_params_t));
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    st = ucc_init(&lib_params, lib_config, &lib);
    ucc_lib_config_release(lib_config);

    st = ucc_context_config_read(lib, NULL, &context_config);
    if (st != UCC_OK) {
        ucc_finalize(lib);
        std::cerr << "failed to read UCC context config: "
                  << ucc_status_string(st);
        return st;
    }
    std::memset(&context_params, 0, sizeof(ucc_context_params_t));
    context_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE |
                          UCC_CONTEXT_PARAM_FIELD_OOB;
    context_params.type = UCC_CONTEXT_SHARED;
    context_params.oob  = bootstrap->get_context_oob();
    ucc_context_create(lib, &context_params, context_config, &context);
    ucc_context_config_release(context_config);
    if (st != UCC_OK) {
        ucc_finalize(lib);
        std::cerr << "failed to create UCC context: " << ucc_status_string(st);
        return st;
    }
    team_params.mask     = UCC_TEAM_PARAM_FIELD_EP |
                           UCC_TEAM_PARAM_FIELD_EP_RANGE |
                           UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob      = bootstrap->get_team_oob();
    team_params.ep       = bootstrap->get_rank();
    team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    st = ucc_team_create_post(&context, 1, &team_params, &team);
    if (st != UCC_OK) {
        std::cerr << "failed to post UCC team create: " <<
                     ucc_status_string(st);
        ucc_context_destroy(context);
        ucc_finalize(lib);
        return st;
    }
    do {
        st = ucc_team_create_test(team);
    } while(st == UCC_INPROGRESS);
    if (st != UCC_OK) {
        std::cerr << "failed to create UCC team: " << ucc_status_string(st);
        ucc_context_destroy(context);
        ucc_finalize(lib);
        return st;
    }
    return UCC_OK;
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
