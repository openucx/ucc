/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <cstring>
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_alltoallv::ucc_pt_coll_alltoallv(ucc_datatype_t dt,
                                             ucc_memory_type mt,
                                             bool is_inplace,
                                             bool is_persistent,
                                             ucc_pt_map_type_t map_type,
                                             ucc_pt_comm *communicator,
                                             ucc_pt_generator_base *generator)
                                             : ucc_pt_coll(communicator, generator)
{
    size_t src_count_max = generator->get_src_count_max();
    size_t dst_count_max = generator->get_dst_count_max();
    size_t src_count_size = src_count_max * ucc_dt_size(dt);
    size_t dst_count_size = dst_count_max * ucc_dt_size(dt);
    ucc_status_t st;

    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = 0;


    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, dst_count_size, mt),
                exit, st);

    if (!is_inplace) {
        UCCCHECK_GOTO(ucc_pt_alloc(&src_header, src_count_size, mt),
                      exit, st);
    }

    coll_args.mask                = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_args.coll_type           = UCC_COLL_TYPE_ALLTOALLV;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
    coll_args.dst.info_v.buffer   = dst_header->addr;
    coll_args.flags               = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                    UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    if (is_inplace) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    } else {
        coll_args.src.info_v.buffer   = src_header->addr;
        coll_args.src.info_v.datatype = dt;
        coll_args.src.info_v.mem_type = mt;

    }

    if (map_type == UCC_PT_MAP_TYPE_LOCAL) {
        ucc_context_h        ctx = comm->get_context();
        ucc_mem_map_t        segments[1];
        ucc_mem_map_params_t mem_map_params;
        size_t               dst_memh_size, src_memh_size;

        mem_map_params.n_segments = 1;
        mem_map_params.segments   = segments;

        mem_map_params.segments[0].address = dst_header->addr;
        mem_map_params.segments[0].len     = dst_count_size;
        UCCCHECK_GOTO(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT,
                                  &mem_map_params, &dst_memh_size, &dst_memh),
                      exit, st);
        coll_args.dst_memh.local_memh = dst_memh;
        coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;

        if (!is_inplace) {
            mem_map_params.segments[0].address = src_header->addr;
            mem_map_params.segments[0].len     = src_count_size;
            UCCCHECK_GOTO(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT,
                                      &mem_map_params, &src_memh_size, &src_memh),
                          exit, st);
            coll_args.src_memh.local_memh = src_memh;
            coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH;
        }
    } else if (map_type == UCC_PT_MAP_TYPE_GLOBAL) {
        ucc_context_h        ctx = comm->get_context();
        ucc_mem_map_t        segments[1];
        ucc_mem_map_params_t mem_map_params;
        uint64_t             dst_memh_size, src_memh_size;
        uint64_t             dst_memh_size_max, src_memh_size_max;
        size_t               memh_size_tmp;
        bool                 map_failed = false;

        coll_args.flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
        mem_map_params.n_segments = 1;
        mem_map_params.segments   = segments;

        mem_map_params.segments[0].address = dst_header->addr;
        mem_map_params.segments[0].len     = dst_count_size;
        st = ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT, &mem_map_params,
                         &memh_size_tmp, &dst_memh);
        if (st != UCC_OK) {
            map_failed    = true;
            dst_memh      = NULL;
            dst_memh_size = 0;
        } else {
            dst_memh_size = (uint64_t)memh_size_tmp;
        }

        comm->allreduce(&dst_memh_size, &dst_memh_size_max, 1, UCC_OP_MAX,
                        UCC_DT_UINT64);

        dst_memh_global = new ucc_mem_map_mem_h[comm->get_size()];
        for (int i = 0; i < comm->get_size(); i++) {
            dst_memh_global[i] = ucc_malloc(dst_memh_size_max, "dst memh blob");
            if (i == comm->get_rank()) {
                if (dst_memh) {
                    memcpy(dst_memh_global[i], dst_memh, dst_memh_size);
                } else {
                    memset(dst_memh_global[i], 0, dst_memh_size_max);
                }
            }
            comm->bcast(dst_memh_global[i], dst_memh_size_max, i);
        }
        for (int i = 0; i < comm->get_size(); i++) {
            st = ucc_mem_map(ctx, UCC_MEM_MAP_MODE_IMPORT, &mem_map_params,
                             &memh_size_tmp, &dst_memh_global[i]);
            if (st != UCC_OK) {
                /* Import failed: dst_memh_global[i] is still a raw serialized
                 * blob, not a valid handle. Free it and any not-yet-imported
                 * entries so the error path only unmaps valid handles. */
                for (int j = i; j < comm->get_size(); j++) {
                    ucc_free(dst_memh_global[j]);
                    dst_memh_global[j] = NULL;
                }
                map_failed = true;
                break;
            }
        }

        if (!is_inplace) {
            mem_map_params.segments[0].address = src_header->addr;
            mem_map_params.segments[0].len     = src_count_size;
            st = ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT, &mem_map_params,
                             &memh_size_tmp, &src_memh);
            if (st != UCC_OK) {
                map_failed    = true;
                src_memh      = NULL;
                src_memh_size = 0;
            } else {
                src_memh_size = (uint64_t)memh_size_tmp;
            }

            comm->allreduce(&src_memh_size, &src_memh_size_max, 1, UCC_OP_MAX,
                            UCC_DT_UINT64);

            src_memh_global = new ucc_mem_map_mem_h[comm->get_size()];
            for (int i = 0; i < comm->get_size(); i++) {
                src_memh_global[i] = ucc_malloc(src_memh_size_max, "src memh blob");
                if (i == comm->get_rank()) {
                    if (src_memh) {
                        memcpy(src_memh_global[i], src_memh, src_memh_size);
                    } else {
                        memset(src_memh_global[i], 0, src_memh_size_max);
                    }
                }
                comm->bcast(src_memh_global[i], src_memh_size_max, i);
            }
            for (int i = 0; i < comm->get_size(); i++) {
                st = ucc_mem_map(ctx, UCC_MEM_MAP_MODE_IMPORT, &mem_map_params,
                                 &memh_size_tmp, &src_memh_global[i]);
                if (st != UCC_OK) {
                    /* Import failed: src_memh_global[i] is still a raw
                     * serialized blob, not a valid handle. Free it and any
                     * not-yet-imported entries so the error path only unmaps
                     * valid handles. */
                    for (int j = i; j < comm->get_size(); j++) {
                        ucc_free(src_memh_global[j]);
                        src_memh_global[j] = NULL;
                    }
                    map_failed = true;
                    break;
                }
            }
        }

        /* A local export/import failure must not let this rank tear down while
         * peers advance to the next barrier. Synchronize the status across
         * ranks so every rank takes the error path together. */
        {
            uint64_t local_fail = map_failed ? 1 : 0;
            uint64_t any_fail   = 0;
            comm->allreduce(&local_fail, &any_fail, 1, UCC_OP_MAX, UCC_DT_UINT64);
            if (any_fail) {
                goto exit;
            }
        }

        coll_args.dst_memh.global_memh = dst_memh_global;
        coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
        coll_args.flags |= UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;

        if (!is_inplace) {
            coll_args.src_memh.global_memh = src_memh_global;
            coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH;
            coll_args.flags |= UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL;
        }
    } else if (map_type != UCC_PT_MAP_TYPE_NONE) {
        std::cerr << "unsupported map type for perftest alltoallv" << std::endl;
        goto exit;
    }

    if (is_persistent) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

    return;
exit:
    if (src_memh) {
        ucc_mem_unmap(&src_memh);
    }
    if (dst_memh) {
        ucc_mem_unmap(&dst_memh);
    }
    if (src_header) {
        ucc_pt_free(src_header);
        src_header = NULL;
    }
    if (dst_header) {
        ucc_pt_free(dst_header);
        dst_header = NULL;
    }
    if (dst_memh_global) {
        for (int i = 0; i < comm->get_size(); i++) {
            if (dst_memh_global[i]) {
                ucc_mem_unmap(&dst_memh_global[i]);
            }
        }
        delete[] dst_memh_global;
        dst_memh_global = NULL;
    }
    if (src_memh_global) {
        for (int i = 0; i < comm->get_size(); i++) {
            if (src_memh_global[i]) {
                ucc_mem_unmap(&src_memh_global[i]);
            }
        }
        delete[] src_memh_global;
        src_memh_global = NULL;
    }
    throw std::runtime_error("failed to initialize alltoallv arguments");
}

ucc_status_t ucc_pt_coll_alltoallv::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args      = test_args.coll_args;

    args = coll_args;
    args.src.info_v.counts        = (ucc_count_t *) generator->get_src_counts();
    args.src.info_v.displacements = (ucc_aint_t *) generator->get_src_displs();
    args.dst.info_v.counts        = (ucc_count_t *) generator->get_dst_counts();
    args.dst.info_v.displacements = (ucc_aint_t *) generator->get_dst_displs();

    return UCC_OK;
}

float ucc_pt_coll_alltoallv::get_bw(float time_ms, int grsize,
                                    ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float            N    = grsize;
    float            S    = 0;
    size_t src_size = 0, dst_size = 0;


    for (int i = 0; i < grsize; i++) {
        src_size += ucc_coll_args_get_count(&args, args.src.info_v.counts, i);
        dst_size += ucc_coll_args_get_count(&args, args.dst.info_v.counts, i);
    }
    src_size *= ucc_dt_size(args.src.info_v.datatype);
    dst_size *= ucc_dt_size(args.dst.info_v.datatype);
    S = src_size > dst_size ? src_size : dst_size;

    return (S / time_ms) * ((N - 1) / N) / 1000.0;
}

ucc_pt_coll_alltoallv::~ucc_pt_coll_alltoallv()
{
    if (src_memh) {
        ucc_mem_unmap(&src_memh);
    }
    if (dst_memh) {
        ucc_mem_unmap(&dst_memh);
    }
    if (src_header) {
        ucc_pt_free(src_header);
    }
    if (dst_header) {
        ucc_pt_free(dst_header);
    }
    if (dst_memh_global) {
        for (int i = 0; i < comm->get_size(); i++) {
            if (dst_memh_global[i]) {
                /* ucc_mem_unmap frees the ucc_malloc'd blob and nulls it. */
                ucc_mem_unmap(&dst_memh_global[i]);
            }
        }
        delete[] dst_memh_global;
    }
    if (src_memh_global) {
        for (int i = 0; i < comm->get_size(); i++) {
            if (src_memh_global[i]) {
                /* ucc_mem_unmap frees the ucc_malloc'd blob and nulls it. */
                ucc_mem_unmap(&src_memh_global[i]);
            }
        }
        delete[] src_memh_global;
    }
}
