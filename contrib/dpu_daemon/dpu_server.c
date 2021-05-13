/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include "server_ucc.h"
#include "host_channel.h"
#include "ucc/api/ucc.h"

#define CORES 8
#define MAX_THREADS 128
typedef struct {
    pthread_t id;
    int idx, nthreads;
    dpu_ucc_comm_t comm;
    dpu_hc_t *hc;
    unsigned int itt;
} thread_ctx_t;

/* thread accisble data - split reader/writer */
typedef struct {
    volatile unsigned long g_itt;  /* first cache line */
    volatile unsigned long pad[3]; /* pad to 64bytes */
    volatile unsigned long l_itt;  /* second cache line */
    volatile unsigned long pad2[3]; /* pad to 64 bytes */
} thread_sync_t;

static thread_sync_t *thread_sync = NULL;

void *dpu_worker(void *arg)
{
    thread_ctx_t *ctx = (thread_ctx_t*)arg;
    int places = CORES/ctx->nthreads;
    int i = 0, j = 0;

    ucc_coll_req_h request;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    CPU_ZERO(&cpuset);
    
	for (i = 0; i < places; i++) {
		CPU_SET((ctx->idx*places)+i, &cpuset);
	}

    i = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);

    while(1) {
        ctx->itt++;
        if (ctx->idx > 0) {
            while (thread_sync[ctx->idx].g_itt < ctx->itt) {
                /* busy wait */
            }
        }
        else {
            dpu_hc_wait(ctx->hc, ctx->itt);
            for (i = 0; i < ctx->nthreads; i++) {
                thread_sync[i].g_itt++;
            }
        }
    
        int offset, block;
        int count = dpu_hc_get_count_total(ctx->hc);
        int ready = 0;
        int dt_size = dpu_ucc_dt_size(dpu_hc_get_dtype(ctx->hc));

        block = count / ctx->nthreads;
        offset = block * ctx->idx;
        if(ctx->idx < (count % ctx->nthreads)) {
            offset += ctx->idx;
            block++;
        } else {
            offset += (count % ctx->nthreads);
        }
        
        ucc_coll_args_t coll = {
            .mask      = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS,
            .coll_type = UCC_COLL_TYPE_ALLREDUCE,
            .src.info = {
                .buffer   = ctx->hc->mem_segs.put.base + offset * dt_size,
                .count    = block * dt_size,
                .datatype = dpu_hc_get_dtype(ctx->hc),
                .mem_type = UCC_MEMORY_TYPE_HOST,
            },
            .dst.info = {
                .buffer     = ctx->hc->mem_segs.get.base + offset * dt_size,
                .count      = block * dt_size,
                .datatype   = dpu_hc_get_dtype(ctx->hc),
                .mem_type = UCC_MEMORY_TYPE_HOST,
            },
            .reduce = {
                .predefined_op = dpu_hc_get_op(ctx->hc),
            },
        };

        if (coll.reduce.predefined_op == UCC_OP_USERDEFINED &&
			coll.src.info.datatype    == UCC_DT_USERDEFINED) {
            break;
        }

        UCC_CHECK(ucc_collective_init(&coll, &request, ctx->comm.team));
        UCC_CHECK(ucc_collective_post(request));
        while (UCC_OK != ucc_collective_test(request)) {
            ucc_context_progress(ctx->comm.ctx);
        }
        UCC_CHECK(ucc_collective_finalize(request));

        thread_sync[ctx->idx].l_itt++;

        if (ctx->idx == 0) {
            while (ready != ctx->nthreads) {
                ready = 0;
                for (i = 0; i < ctx->nthreads; i++) {
                    if (thread_sync[i].l_itt == ctx->itt) {
                        ready++;
                    }
                    else {
                        break;
                    }
                }
            }
    
            dpu_hc_reply(ctx->hc, ctx->itt);
        }
    }

//     fprintf(stderr, "ctx->itt = %u\n", ctx->itt);

    return NULL;
}

int main(int argc, char **argv)
{
//     fprintf (stderr, "%s\n", __FUNCTION__);
//     sleep(20);

    int nthreads = 0, i;
    thread_ctx_t *tctx_pool = NULL;
    dpu_ucc_global_t ucc_glob;
    dpu_hc_t hc_b, *hc = &hc_b;

    if (argc < 2 ) {
        printf("Need thread # as an argument\n");
        return 1;
    }
    nthreads = atoi(argv[1]);
    if (MAX_THREADS < nthreads || 0 >= nthreads) {
        printf("ERROR: bad thread #: %d\n", nthreads);
        return 1;
    }
    printf("DPU daemon: Running with %d threads\n", nthreads);
    tctx_pool = calloc(nthreads, sizeof(*tctx_pool));
    UCC_CHECK(dpu_ucc_init(argc, argv, &ucc_glob));

//     thread_sync = calloc(nthreads, sizeof(*thread_sync));
    thread_sync = aligned_alloc(64, nthreads * sizeof(*thread_sync));
    memset(thread_sync, 0, nthreads * sizeof(*thread_sync));

    dpu_hc_init(hc);
    dpu_hc_accept(hc);

    for(i = 0; i < nthreads; i++) {
//         printf("Thread %d spawned!\n", i);
        UCC_CHECK(dpu_ucc_alloc_team(&ucc_glob, &tctx_pool[i].comm));
        tctx_pool[i].idx = i;
        tctx_pool[i].nthreads = nthreads;
        tctx_pool[i].hc       = hc;
        tctx_pool[i].itt = 0;

        if (i < nthreads - 1) {
            pthread_create(&tctx_pool[i].id, NULL, dpu_worker,
                           (void*)&tctx_pool[i]);
        }
    }

    /* The final DPU worker is executed in this context */
    dpu_worker((void*)&tctx_pool[i-1]);

    for(i = 0; i < nthreads; i++) {
        if (i < nthreads - 1) {
            pthread_join(tctx_pool[i].id, NULL);
        }
        dpu_ucc_free_team(&ucc_glob, &tctx_pool[i].comm);
//         printf("Thread %d joined!\n", i);
    }

    dpu_ucc_finalize(&ucc_glob);
    return 0;
}
