/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <iomanip>
#include "ucc_pt_benchmark.h"
#include "components/mc/ucc_mc.h"
#include "ucc_perftest.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_ee.h"
#include "ucc_pt_coll.h"
#include "generator/ucc_pt_generator.h"

ucc_pt_benchmark::ucc_pt_benchmark(ucc_pt_benchmark_config cfg,
                                   ucc_pt_comm *communicator):
    config(cfg),
    comm(communicator)
{
    /* RNG seed is passed to generators for isolated reproducibility */

    if (cfg.gen.type == UCC_PT_GEN_TYPE_EXP) {
        generator = new ucc_pt_generator_exponential(cfg.min_count, cfg.max_count, 2,
                                                     communicator->get_size(),
                                                     cfg.op_type);
    } else if (cfg.gen.type == UCC_PT_GEN_TYPE_FILE) {
        generator = new ucc_pt_generator_file(cfg.gen.file_name,
                                              communicator->get_size(),
                                              communicator->get_rank(),
                                              cfg.op_type, cfg.gen.nrep);
        if (cfg.op_type != UCC_PT_OP_TYPE_ALLTOALLV) {
            throw std::runtime_error("Only ALLTOALLV is supported for file generator");
        }
    } else if (cfg.gen.type == UCC_PT_GEN_TYPE_TRAFFIC_MATRIX) {
        if (cfg.op_type != UCC_PT_OP_TYPE_ALLTOALLV) {
            throw std::runtime_error(
                "Only ALLTOALLV is supported for traffic matrix generator");
        }
        generator = new ucc_pt_generator_traffic_matrix(
            cfg.gen.matrix.kind,
            communicator->get_size(),
            communicator->get_rank(),
            cfg.dt,
            cfg.op_type,
            cfg.gen.nrep,
            cfg.gen.matrix.token_size_KB_mean,
            cfg.gen.matrix.num_tokens,
            cfg.gen.matrix.tgt_group_size_mean,
            cfg.seed);
    } else {
        /* assuming that the generator type is UCC_PT_GEN_TYPE_EXP */
        generator = new ucc_pt_generator_exponential(cfg.min_count, cfg.max_count, 2,
                                                     communicator->get_size(),
                                                     cfg.op_type);
    }

    switch (cfg.op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
        coll = new ucc_pt_coll_allgather(cfg.dt, cfg.mt, cfg.inplace,
                                         cfg.persistent, cfg.map_type,
                                         comm, generator);
        break;
    case UCC_PT_OP_TYPE_ALLGATHERV:
        coll = new ucc_pt_coll_allgatherv(cfg.dt, cfg.mt, cfg.inplace,
                                          cfg.persistent, comm, generator);
        break;
    case UCC_PT_OP_TYPE_ALLREDUCE:
        coll = new ucc_pt_coll_allreduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                         cfg.persistent, comm, generator);
        break;
    case UCC_PT_OP_TYPE_ALLTOALL:
        coll = new ucc_pt_coll_alltoall(cfg.dt, cfg.mt, cfg.inplace,
                                        cfg.persistent, cfg.map_type,
                                        comm, generator);
        break;
    case UCC_PT_OP_TYPE_ALLTOALLV:
        coll = new ucc_pt_coll_alltoallv(cfg.dt, cfg.mt, cfg.inplace,
                                         cfg.persistent, comm, generator);
        break;
    case UCC_PT_OP_TYPE_BARRIER:
        coll = new ucc_pt_coll_barrier(comm, generator);
        break;
    case UCC_PT_OP_TYPE_BCAST:
        coll = new ucc_pt_coll_bcast(cfg.dt, cfg.mt, cfg.root_shift,
                                     cfg.persistent, comm, generator);
        break;
    case UCC_PT_OP_TYPE_GATHER:
        coll = new ucc_pt_coll_gather(cfg.dt, cfg.mt, cfg.inplace,
                                      cfg.persistent, cfg.root_shift, comm, generator);
        break;
    case UCC_PT_OP_TYPE_GATHERV:
        coll = new ucc_pt_coll_gatherv(cfg.dt, cfg.mt, cfg.inplace,
                                       cfg.persistent, cfg.root_shift, comm, generator);
        break;
    case UCC_PT_OP_TYPE_REDUCE:
        coll = new ucc_pt_coll_reduce(cfg.dt, cfg.mt, cfg.op, cfg.inplace,
                                      cfg.persistent, cfg.root_shift, comm, generator);
        break;
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
        coll = new ucc_pt_coll_reduce_scatter(cfg.dt, cfg.mt, cfg.op,
                                              cfg.inplace,
                                              cfg.persistent, comm, generator);
        break;
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
        coll = new ucc_pt_coll_reduce_scatterv(cfg.dt, cfg.mt, cfg.op,
                                               cfg.inplace, cfg.persistent,
                                               comm, generator);
        break;
    case UCC_PT_OP_TYPE_SCATTER:
        coll = new ucc_pt_coll_scatter(cfg.dt, cfg.mt, cfg.inplace,
                                       cfg.persistent, cfg.root_shift, comm, generator);
        break;
    case UCC_PT_OP_TYPE_SCATTERV:
        coll = new ucc_pt_coll_scatterv(cfg.dt, cfg.mt, cfg.inplace,
                                        cfg.persistent, cfg.root_shift, comm, generator);
        break;
    case UCC_PT_OP_TYPE_MEMCPY:
        coll = new ucc_pt_op_memcpy(cfg.dt, cfg.mt, cfg.n_bufs, comm, generator);
        break;
    case UCC_PT_OP_TYPE_REDUCEDT:
        coll = new ucc_pt_op_reduce(cfg.dt, cfg.mt, cfg.op, cfg.n_bufs, comm, generator);
        break;
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
        coll = new ucc_pt_op_reduce_strided(cfg.dt, cfg.mt, cfg.op, cfg.n_bufs,
                                            comm, generator);
        break;
    default:
        throw std::runtime_error("not supported collective");
    }
}

ucc_status_t ucc_pt_benchmark::run_bench() noexcept
{
    ucc_status_t       st;
    ucc_pt_test_args_t args;
    double             time;
    double             time_min, time_max, time_avg;
    double             total_time = 0;

    generator->reset();
    print_header();


    for (generator->reset(); generator->has_next(); generator->next()) {
        int iter = config.n_iter_small;
        int warmup = config.n_warmup_small;

        if (generator->get_count_max() >= config.large_thresh) {
            iter = config.n_iter_large;
            warmup = config.n_warmup_large;
        }
        args.coll_args.root = config.root;
        UCCCHECK_GOTO(coll->init_args(args), exit_err, st);
        if ((uint64_t)config.op_type < (uint64_t)UCC_COLL_TYPE_LAST) {
            UCCCHECK_GOTO(run_single_coll_test(args.coll_args, warmup, iter, time),
                          free_coll, st);
        } else {
            UCCCHECK_GOTO(run_single_executor_test(args.executor_args,
                                                   warmup, iter, time),
                          free_coll, st);
        }

        comm->allreduce(&time, &time_min, 1, UCC_OP_MIN, UCC_DT_FLOAT64);
        comm->allreduce(&time, &time_max, 1, UCC_OP_MAX, UCC_DT_FLOAT64);
        comm->allreduce(&time, &time_avg, 1, UCC_OP_SUM, UCC_DT_FLOAT64);
        time_avg /= comm->get_size();
        total_time += time_max;

        print_time(generator->get_src_count(), args, time_avg, time_min, time_max);
        coll->free_args(args);
        if (!coll->has_range()) {
            /* exit here since collective doesn't have count argument */
            break;
        }
    }

    if (comm->get_rank() == 0) {
        std::cout << "Total time: " << total_time / 1000 << " ms" << std::endl;
    }

    return UCC_OK;
free_coll:
    coll->free_args(args);
exit_err:
    return st;
}

static inline double get_time_us(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

ucc_status_t ucc_pt_benchmark::run_single_coll_test(ucc_coll_args_t args,
                                                    int nwarmup, int niter,
                                                    double &time)
                                                    noexcept
{
    const bool    triggered  = config.triggered;
    const bool    persistent = config.persistent;
    ucc_team_h    team       = comm->get_team();
    ucc_context_h ctx        = comm->get_context();
    ucc_status_t  st         = UCC_OK;
    ucc_coll_req_h req;
    ucc_ee_h ee;
    ucc_ev_t comp_ev, *post_ev;

    UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    time = 0;

    if (triggered) {
        try {
            ee = comm->get_ee();
        } catch(std::exception &e) {
            std::cerr << e.what() << std::endl;
            return UCC_ERR_NO_MESSAGE;
        }
        /* dummy event, for benchmark purposes no real event required */
        comp_ev.ev_type         = UCC_EVENT_COMPUTE_COMPLETE;
        comp_ev.ev_context      = nullptr;
        comp_ev.ev_context_size = 0;
    }

    if (persistent) {
        UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
    }

    args.root = config.root % comm->get_size();
    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();

        if (!persistent) {
            UCCCHECK_GOTO(ucc_collective_init(&args, &req, team), exit_err, st);
        }

        if (triggered) {
            comp_ev.req = req;
            UCCCHECK_GOTO(ucc_collective_triggered_post(ee, &comp_ev),
                          free_req, st);
            UCCCHECK_GOTO(ucc_ee_get_event(ee, &post_ev), free_req, st);
            ucc_assert(post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
            UCCCHECK_GOTO(ucc_ee_ack_event(ee, post_ev), free_req, st);
        } else {
            UCCCHECK_GOTO(ucc_collective_post(req), free_req, st);
        }

        st = ucc_collective_test(req);
        while (st > 0) {
            UCCCHECK_GOTO(ucc_context_progress(ctx), free_req, st);
            st = ucc_collective_test(req);
        }

        if (!persistent) {
            ucc_collective_finalize(req);
        }
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
        args.root = (args.root + config.root_shift) % comm->get_size();
        UCCCHECK_GOTO(comm->barrier(), exit_err, st);
    }

    if (persistent) {
        ucc_collective_finalize(req);
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

ucc_status_t
ucc_pt_benchmark::run_single_executor_test(ucc_ee_executor_task_args_t args,
                                           int nwarmup, int niter,
                                           double &time) noexcept
{
    const bool              triggered = config.triggered;
    ucc_ee_executor_t      *executor  = comm->get_executor();
    ucc_status_t            st        = UCC_OK;
    ucc_ee_h                ee;
    ucc_ee_executor_task_t *task;

    time = 0;
    if (triggered) {
        try {
            ee = comm->get_ee();
        } catch(std::exception &e) {
            std::cerr << e.what() << std::endl;
            return UCC_ERR_NO_MESSAGE;
        }
        UCCCHECK_GOTO(ucc_ee_executor_start(executor, ee->ee_context),
                      exit_err, st);
    } else {
        UCCCHECK_GOTO(ucc_ee_executor_start(executor, nullptr), exit_err, st);
    }

    for (int i = 0; i < nwarmup + niter; i++) {
        double s = get_time_us();

        UCCCHECK_GOTO(ucc_ee_executor_task_post(executor, &args, &task),
                      stop_exec, st);
        st = ucc_ee_executor_task_test(task);
        while (st > 0) {
            st = ucc_ee_executor_task_test(task);
        }
        ucc_ee_executor_task_finalize(task);
        double f = get_time_us();
        if (st != UCC_OK) {
            goto exit_err;
        }
        if (i >= nwarmup) {
            time += f - s;
        }
    }

    UCCCHECK_GOTO(ucc_ee_executor_stop(executor), exit_err, st);
    if (niter != 0) {
        time /= niter;
    }
    return UCC_OK;

stop_exec:
    ucc_ee_executor_stop(executor);
exit_err:
    return st;
}

void ucc_pt_benchmark::print_header()
{
    if (comm->get_rank() == 0) {
        std::ios iostate(nullptr);
        iostate.copyfmt(std::cout);
        std::cout << std::left << std::setw(24)
                  << "Collective: " << ucc_pt_op_type_str(config.op_type)
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
            std::cout << std::setw(42) << "Bus Bandwidth, GB/s";
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

void ucc_pt_benchmark::print_time(size_t count, ucc_pt_test_args_t args,
                                  double time_avg,
                                  double time_min,
                                  double time_max)
{
    size_t size    = count * ucc_dt_size(config.dt);
    int    gsize   = comm->get_size();

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
                if (config.op_type == UCC_PT_OP_TYPE_GATHER ||
                    config.op_type == UCC_PT_OP_TYPE_SCATTER) {
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
    delete generator;
}
