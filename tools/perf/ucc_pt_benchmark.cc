#include <iomanip>
#include "ucc_pt_benchmark.h"
#include "core/ucc_mc.h"
#include "ucc_perftest.h"
#include "utils/ucc_coll_utils.h"

ucc_pt_benchmark::ucc_pt_benchmark(ucc_pt_benchmark_config cfg,
                                   ucc_pt_comm *communcator):
    config(cfg),
    comm(communcator)
{
    switch (cfg.coll_type) {
    case UCC_COLL_TYPE_ALLGATHER:
        coll = new ucc_pt_coll_allgather(comm->get_size(), cfg.dt, cfg.mt,
                                         cfg.inplace);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        coll = new ucc_pt_coll_alltoall(comm->get_size(), cfg.dt, cfg.mt,
                                        cfg.inplace);
        break;
    case UCC_COLL_TYPE_BARRIER:
        coll = new ucc_pt_coll_barrier();
        break;
    case UCC_COLL_TYPE_BCAST:
        coll = new ucc_pt_coll_bcast(cfg.dt, cfg.mt);
        break;
    default:
        throw std::runtime_error("not supported collective");
    }
}

ucc_status_t ucc_pt_benchmark::run_bench() noexcept
{
    size_t min_count = coll->has_range() ? config.min_count : 1;
    size_t max_count = coll->has_range() ? config.max_count : 1;
    ucc_status_t st;
    ucc_coll_args_t args;
    std::chrono::nanoseconds time;

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
        UCCCHECK_GOTO(run_single_test(args, iter, warmup, time), free_coll, st);
        coll->free_coll_args(args);
        print_time(cnt, time);
    }

    return UCC_OK;
free_coll:
    coll->free_coll_args(args);
exit_err:
    return st;
}

ucc_status_t ucc_pt_benchmark::run_single_test(ucc_coll_args_t args,
                                               int nwarmup, int niter,
                                               std::chrono::nanoseconds &time)
                                               noexcept
{
    ucc_team_h    team = comm->get_team();
    ucc_context_h ctx  = comm->get_context();
    ucc_status_t  st   = UCC_OK;
    ucc_coll_req_h req;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = std::chrono::nanoseconds::zero();
    for (int i = 0; i < nwarmup + niter; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
        UCCCHECK_GOTO(ucc_collective_post(req), free_req, st);
        st = ucc_collective_test(req);
        while (st == UCC_INPROGRESS) {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
            st = ucc_collective_test(req);
        }
        ucc_collective_finalize(req);
        auto f = std::chrono::high_resolution_clock::now();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(f - s);
        }
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }
    time /= niter;
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
                  << std::setw(24) << "Time, us"
                  << std::endl;
        std::cout << std::setw(36) << "avg" <<
                     std::setw(12) << "min" <<
                     std::setw(12) << "max" <<
                     std::endl;
    }
}

void ucc_pt_benchmark::print_time(size_t count, std::chrono::nanoseconds time)
{
    float  time_ms = time.count() / 1000.0;
    size_t size    = count * ucc_dt_size(config.dt);
    float time_avg, time_min, time_max;

    comm->allreduce(&time_ms, &time_min, 1, UCC_OP_MIN);
    comm->allreduce(&time_ms, &time_max, 1, UCC_OP_MAX);
    comm->allreduce(&time_ms, &time_avg, 1, UCC_OP_SUM);
    time_avg /= comm->get_size();

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
                  << std::setw(12) << time_max
                  << std::endl;
        std::cout.copyfmt(iostate);
    }
}

ucc_pt_benchmark::~ucc_pt_benchmark()
{
    delete coll;
}
