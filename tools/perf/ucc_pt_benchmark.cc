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
    coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace);
}

ucc_status_t ucc_pt_benchmark::run_bench()
{
    ucc_status_t st = UCC_OK;
    ucc_coll_args_t args;
    std::chrono::nanoseconds time;

    print_header();
    for (size_t cnt = config.min_count; cnt <= config.max_count; cnt *= 2) {
        UCCCHECK_GOTO(coll->get_coll(cnt, args), exit_err, st);
        UCCCHECK_GOTO(run_single_test(args, time), free_coll, st);
        coll->free_coll(args);
        print_time(cnt, time);
    }

    return UCC_OK;
free_coll:
    coll->free_coll(args);
exit_err:
    return st;
}

ucc_status_t ucc_pt_benchmark::run_single_test(ucc_coll_args_t args,
                                               std::chrono::nanoseconds &time)
{
    ucc_team_h    team = comm->get_team();
    ucc_context_h ctx  = comm->get_context();
    ucc_status_t  st   = UCC_OK;
    ucc_coll_req_h req;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = std::chrono::nanoseconds::zero();
    for (int i = 0; i < config.n_warmup + config.n_iter; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
        UCCCHECK_GOTO(ucc_collective_post(req), free_req, st);
        do {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
        } while (ucc_collective_test(req) == UCC_INPROGRESS);
        ucc_collective_finalize(req);
        auto f = std::chrono::high_resolution_clock::now();
        if (i >= config.n_warmup) {
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(f - s);
        }
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }
    time /= config.n_iter;
    return UCC_OK;
free_req:
    ucc_collective_finalize(req);
exit_err:
    return st;
}

void ucc_pt_benchmark::print_header()
{
    if (comm->get_rank() == 0) {
        std::cout << "Collective: " << ucc_coll_type_str(config.coll_type)
                  << std::endl;
        std::cout << "Memory type: " << ucc_memory_type_names[config.mt]
                  << std::endl;
        std::cout << "Data type: " << config.dt
                  << std::endl;
        std::cout << "Operation type: " << config.op
                  << std::endl;
        std::cout << "Warmup: "<< config.n_warmup << "; "
                     "Iterations: "<< config.n_iter
                  << std::endl;
        std::cout << std::setw(12) << "Count" <<
                     std::setw(12) << "Size" <<
                     std::setw(24) << "Time, us" <<
                     std::endl;
        std::cout << std::setw(36) << "avg" <<
                     std::setw(12) << "min" <<
                     std::setw(12) << "max" <<
                     std::endl;
    }
}

void ucc_pt_benchmark::print_time(size_t count, std::chrono::nanoseconds time)
{
    float time_ms = time.count() / 1000.0;
    float time_avg, time_min, time_max;
    comm->allreduce(&time_ms, &time_min, 1, UCC_OP_MIN);
    comm->allreduce(&time_ms, &time_max, 1, UCC_OP_MAX);
    comm->allreduce(&time_ms, &time_avg, 1, UCC_OP_SUM);
    time_avg /= comm->get_size();

    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout<<std::setprecision(2) << std::fixed;
        std::cout << std::setw(12) << count <<
                     std::setw(12) << count * ucc_dt_size(config.dt) <<
                     std::setw(12) << time_avg <<
                     std::setw(12) << time_min <<
                     std::setw(12) << time_max <<
                     std::endl;
        std::cout.copyfmt(iostate);
    }
}

ucc_pt_benchmark::~ucc_pt_benchmark()
{
    delete coll;
}
