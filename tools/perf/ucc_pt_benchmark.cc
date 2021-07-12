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
    case UCC_COLL_TYPE_ALLGATHERV:
        coll = new ucc_pt_coll_allgatherv(comm->get_size(), cfg.dt, cfg.mt,
                                          cfg.inplace);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        coll = new ucc_pt_coll_alltoall(comm->get_size(), cfg.dt, cfg.mt,
                                        cfg.inplace);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        coll = new ucc_pt_coll_alltoallv(comm->get_size(), cfg.dt, cfg.mt,
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

typedef struct test_args {
    ucc_pt_comm *    comm;
    ucc_coll_args_t *args;
    int              nwarmup;
    int              niter;
} test_args_t;

typedef struct thread_args {
    test_args_t *            test_args;
    std::chrono::nanoseconds time;
    int                      thread_id;
    ucc_status_t             st;
} thread_args_t;

void *run_single_test(void *arg);

ucc_status_t ucc_pt_benchmark::run_bench() noexcept
{
    size_t                 min_count = coll->has_range() ? config.min_count : 1;
    size_t                 max_count = coll->has_range() ? config.max_count : 1;
    ucc_status_t           st;
    ucc_coll_args_t        args;
    test_args_t            test_args;
    int                    i, n_threads = comm->get_n_threads();
    std::vector<pthread_t> threads;
    threads.resize(n_threads);
    std::vector<thread_args_t> threads_args;
    threads_args.resize(n_threads);

    print_header();
    for (size_t cnt = min_count; cnt <= max_count; cnt *= 2) {
        size_t coll_size = cnt * ucc_dt_size(config.dt);
        int    iter      = config.n_iter_small;
        int    warmup    = config.n_warmup_small;
        if (coll_size >= config.large_thresh) {
            iter   = config.n_iter_large;
            warmup = config.n_warmup_large;
        }
        UCCCHECK_GOTO(coll->init_coll_args(cnt, args), exit_err, st);
        test_args = (test_args_t){comm, &args, warmup, iter};
        if (n_threads > 1) {
            for (i = 0; i < n_threads; i++) {
                threads_args.at(i).thread_id = i;
                threads_args.at(i).test_args = &test_args;
                pthread_create(&threads[i], NULL, &run_single_test,
                               (void *)&threads_args[i]);
            }
            for (i = 0; i < n_threads; i++) {
                pthread_join(threads[i], NULL);
                UCCCHECK_GOTO(threads_args[i].st, free_coll, st);
            }
        }
        else {
            threads_args[0].thread_id = 0;
            threads_args[0].test_args = &test_args;
            run_single_test((void *)&threads_args[0]);
            UCCCHECK_GOTO(threads_args[0].st, free_coll, st);
        }
        coll->free_coll_args(args);
        for (i = 1; i < n_threads; i++) {
            threads_args[0].time += threads_args[i].time;
        }
        threads_args[0].time /= n_threads;
        print_time(cnt, threads_args[0].time);
    }
    return UCC_OK;
free_coll:
    coll->free_coll_args(args);
exit_err:
    return st;
}

void *run_single_test(void *arg)

{
    thread_args_t *thread_args = (thread_args_t *)arg;
    ucc_team_h     team =
        thread_args->test_args->comm->get_team(thread_args->thread_id);
    ucc_context_h  ctx  = thread_args->test_args->comm->get_context();
    ucc_status_t  st   = UCC_OK;
    ucc_coll_req_h req;

    UCCCHECK_GOTO(thread_args->test_args->comm->barrier(thread_args->thread_id),
                  exit_err, st);
    thread_args->time = std::chrono::nanoseconds::zero();
    for (int i = 0; i < thread_args->test_args->nwarmup +
         thread_args->test_args->niter; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        UCCCHECK_GOTO(
            ucc_collective_init(thread_args->test_args->args, &req, team),
            exit_err, st);
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
        if (i >= thread_args->test_args->nwarmup) {
            thread_args->time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(f - s);
        }
        UCCCHECK_GOTO(
            thread_args->test_args->comm->barrier(thread_args->thread_id),
            exit_err, st);
    }
    if (thread_args->test_args->niter) {
        thread_args->time /= thread_args->test_args->niter;
    }
    thread_args->st = st;
    return 0;
free_req:
    ucc_collective_finalize(req);
exit_err:
    thread_args->st = st;
    return 0;
}

void ucc_pt_benchmark::print_header()
{
    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::left << std::setw(24)
                  << "Number of threads: " << std::to_string(config.n_threads)
                  << std::endl;
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
