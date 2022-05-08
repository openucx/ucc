#include <iomanip>
#include "ucc_pt_benchmark.h"
#include "components/mc/ucc_mc.h"
#include "ucc_perftest.h"
#include "utils/ucc_coll_utils.h"

ucc_pt_benchmark::ucc_pt_benchmark(ucc_pt_benchmark_config cfg,
                                   ucc_pt_comm *communicator):
    config(cfg),
    comm(communicator)
{
    switch (cfg.coll_type) {
    case UCC_COLL_TYPE_ALLGATHER:
        coll = new ucc_pt_coll_allgather(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        coll = new ucc_pt_coll_allgatherv(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                         comm);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        coll = new ucc_pt_coll_alltoall(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        coll = new ucc_pt_coll_alltoallv(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_BARRIER:
        coll = new ucc_pt_coll_barrier(comm);
        break;
    case UCC_COLL_TYPE_BCAST:
        coll = new ucc_pt_coll_bcast(cfg.dt, cfg.mt, comm);
        break;
    case UCC_COLL_TYPE_REDUCE:
        coll = new ucc_pt_coll_reduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                      comm);
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        coll = new ucc_pt_coll_reduce_scatter(cfg.dt, cfg.mt, cfg.op,
                                              cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_GATHER:
        coll = new ucc_pt_coll_gather(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_GATHERV:
        coll = new ucc_pt_coll_gatherv(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_SCATTER:
        coll = new ucc_pt_coll_scatter(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    case UCC_COLL_TYPE_SCATTERV:
        coll = new ucc_pt_coll_scatterv(cfg.dt, cfg.mt, cfg.inplace, comm);
        break;
    default:
        throw std::runtime_error("not supported collective");
    }
}

ucc_status_t ucc_pt_benchmark::run_bench() noexcept
{
    size_t min_count = coll->has_range() ? config.min_count : 1;
    size_t max_count = coll->has_range() ? config.max_count : 1;
    ucc_status_t    st;
    ucc_coll_args_t args;
    double          time;

    print_header();
    for (size_t cnt = min_count; cnt <= max_count; cnt *= 2) {
        size_t coll_size = cnt * ucc_dt_size(config.dt);
        int iter = config.n_iter_small;
        int warmup = config.n_warmup_small;
        if (coll_size >= config.large_thresh) {
            iter = config.n_iter_large;
            warmup = config.n_warmup_large;
        }
        UCCCHECK_GOTO(coll->init_coll_args(cnt, args), exit_err, st);
        UCCCHECK_GOTO(run_single_test(args, warmup, iter, time), free_coll, st);
        print_time(cnt, args, time);
        coll->free_coll_args(args);
    }

    return UCC_OK;
free_coll:
    coll->free_coll_args(args);
exit_err:
    return st;
}

static inline double get_time_us(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

ucc_status_t ucc_pt_benchmark::run_single_test(ucc_coll_args_t args,
                                               int nwarmup, int niter,
                                               double &time)
                                               noexcept
{
    ucc_team_h    team = comm->get_team();
    ucc_context_h ctx  = comm->get_context();
    ucc_status_t  st   = UCC_OK;
    ucc_coll_req_h req;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = 0;

    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();
        UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
        UCCCHECK_GOTO(ucc_collective_post(req), free_req, st);
        st = ucc_collective_test(req);
        while (st == UCC_INPROGRESS) {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
            st = ucc_collective_test(req);
        }
        ucc_collective_finalize(req);
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }
    if (niter != 0) {
        time /= niter;
    }
    return UCC_OK;
free_req:
    ucc_collective_finalize(req);
exit_err:
    return st;
}

void ucc_pt_benchmark::print_header()
{
    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::left << std::setw(24)
                  << "Collective: " << ucc_coll_type_str(config.coll_type)
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Memory type: " << ucc_memory_type_names[config.mt]
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Datatype: " << ucc_datatype_str(config.dt)
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Reduction: "
                  << (coll->has_reduction() ?
                        ucc_reduction_op_str(config.op):
                        "N/A")
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Inplace: "
                  << (coll->has_inplace() ?
                        std::to_string(config.inplace):
                        "N/A")
                  << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Warmup:" << std::endl
                  << std::left << std::setw(24)
                  << "  small" << config.n_warmup_small << std::endl
                  << std::left << std::setw(24)
                  << "  large" << config.n_warmup_large << std::endl;
        std::cout << std::left << std::setw(24)
                  << "Iterations:" << std::endl
                  << std::left << std::setw(24)
                  << "  small" << config.n_iter_small << std::endl
                  << std::left << std::setw(24)
                  << "  large" << config.n_iter_large << std::endl;
        std::cout.copyfmt(iostate);
        std::cout << std::endl;
        std::cout << std::setw(12) << "Count"
                  << std::setw(12) << "Size"
                  << std::setw(24) << "Time, us";
        if (config.full_print) {
            std::cout << std::setw(42) << "Bandwidth, GB/s";
        }
        std::cout << std::endl;
        std::cout << std::setw(36) << "avg"
                  << std::setw(12) << "min"
                  << std::setw(12) << "max";
        if (config.full_print) {
            std::cout << std::setw(12) << "avg"
                      << std::setw(12) << "max"
                      << std::setw(12) << "min";
        }
        std::cout << std::endl;
    }
}

void ucc_pt_benchmark::print_time(size_t count, ucc_coll_args_t args,
                                  double time)
{
    double time_us = time;
    size_t size    = count * ucc_dt_size(config.dt);
    int    gsize   = comm->get_size();
    double time_avg, time_min, time_max;

    comm->allreduce(&time_us, &time_min, 1, UCC_OP_MIN);
    comm->allreduce(&time_us, &time_max, 1, UCC_OP_MAX);
    comm->allreduce(&time_us, &time_avg, 1, UCC_OP_SUM);
    time_avg /= gsize;

    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::setprecision(2) << std::fixed;
        std::cout << std::setw(12) << (coll->has_range() ?
                                        std::to_string(count):
                                        "N/A")
                  << std::setw(12) << (coll->has_range() ?
                                        std::to_string(size):
                                        "N/A")
                  << std::setw(12) << time_avg
                  << std::setw(12) << time_min
                  << std::setw(12) << time_max;

        if (config.full_print) {
            if (!coll->has_bw()) {
                std::cout << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A";
            } else {
                if (config.coll_type == UCC_COLL_TYPE_GATHER ||
                    config.coll_type == UCC_COLL_TYPE_SCATTER) {
                    std::cout << std::setw(12) << "N/A"
                              << std::setw(12) << "N/A"
                              << std::setw(12) << coll->get_bw(time_max, gsize,
                                                               args);
                } else {
                    std::cout << std::setw(12) << coll->get_bw(time_avg, gsize,
                                                               args)
                              << std::setw(12) << coll->get_bw(time_min, gsize,
                                                               args)
                              << std::setw(12) << coll->get_bw(time_max, gsize,
                                                               args);
                }
            }
        }
        std::cout << std::endl;
        std::cout.copyfmt(iostate);
    }
}

ucc_pt_benchmark::~ucc_pt_benchmark()
{
    delete coll;
}
