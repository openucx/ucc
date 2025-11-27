/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"
#include "coll_patterns/bruck_alltoall.h"
#include "utils/ucc_atomic.h"

/*
scratch structure
buff_size - alltoallv_hybrid_buff_size
======================================================================================================================================================================================================================
|sdispls            |scounts            |calc                   |seg                 |lb                                                         |ptrs                                      |buf                     |
|tsize * sizeof(int)|tsize * sizeof(int)|2 * tsize * sizeof(int)|tsize * sizeof(char)|sizeof(ucc_tl_ucp_alltoallv_hybrid_buf_meta_t) * (radix -1)|ucc_div_round_up(tsize, 2) * sizeof(void*)|(radix - 1) * buff_size |
======================================================================================================================================================================================================================
*/

#define NUM_BINS          5
#define MAX_BRUCK         1250
#define COUNT_DIRECT      UINT_MAX
#define DENSE_PACK_FORMAT UINT_MAX

enum {
    UCC_ALLTOALLV_HYBRID_PHASE_START,
    UCC_ALLTOALLV_HYBRID_PHASE_SENT,
    UCC_ALLTOALLV_HYBRID_PHASE_RECV
};

enum {
    ALLTOALLV_HYBRID_SEG_SEND_DIRECT = 0,
    ALLTOALLV_HYBRID_SEG_RECV_DIRECT = 1,
    ALLTOALLV_HYBRID_SEG_DIGIT       = 2
};

typedef struct ucc_tl_ucp_alltoallv_hybrid_merge_bin {
    ucc_tl_ucp_task_t *task;
    int                start;
    int                len;
} ucc_tl_ucp_alltoallv_hybrid_merge_bin_t;

typedef struct ucc_tl_ucp_alltoallv_hybrid_buf_meta {
    ucc_tl_ucp_alltoallv_hybrid_merge_bin_t bins[NUM_BINS];
    int                                     cur_bin;
    int                                     offset;
} ucc_tl_ucp_alltoallv_hybrid_buf_meta_t;

#define TASK_SCRATCH(_task) ({                                                 \
             (_task)->alltoallv_hybrid.scratch_mc_header->addr;                \
        })

#define TASK_TMP(_task, _tsize) ({                                             \
            PTR_OFFSET(TASK_SCRATCH(_task), 2*_tsize*sizeof(int));             \
        })

#define TASK_SEG(_task, _tsize) ({                                             \
            PTR_OFFSET(TASK_SCRATCH(_task), 4 * _tsize*sizeof(int));           \
        })

#define TASK_LB(_task, _tsize) ({                                              \
            PTR_OFFSET(TASK_SEG(_task, _tsize), _tsize*sizeof(char));          \
        })

#define TASK_PTRS(_task, _tsize) ({                                                                          \
            uint32_t _radix  = (_task)->alltoallv_hybrid.radix;                                              \
            PTR_OFFSET(TASK_LB(_task, _tsize), sizeof(ucc_tl_ucp_alltoallv_hybrid_buf_meta_t)*(_radix - 1)); \
        })

#define TASK_BUF(_task, _i, _tsize) ({                                                                  \
            size_t _buff_size  = UCC_TL_UCP_TEAM_LIB(TASK_TEAM(_task))->cfg.alltoallv_hybrid_buff_size; \
            PTR_OFFSET(TASK_PTRS(_task, _tsize), ucc_div_round_up(_tsize, 2)*sizeof(void*) +            \
                       (_i) * _buff_size);                                                              \
        })

#define ALIGN(x) ucc_ceil(x, 4)

#define IS_DIRECT_SEND(_seg) ((_seg) & UCC_BIT(ALLTOALLV_HYBRID_SEG_SEND_DIRECT))

#define SET_DIRECT_SEND(_seg) ((_seg) |= UCC_BIT(ALLTOALLV_HYBRID_SEG_SEND_DIRECT))

#define IS_DIRECT_RECV(_seg) ((_seg) & UCC_BIT(ALLTOALLV_HYBRID_SEG_RECV_DIRECT))

#define SET_DIRECT_RECV(_seg) ((_seg) |= UCC_BIT(ALLTOALLV_HYBRID_SEG_RECV_DIRECT))

#define GET_BRUCK_DIGIT(_seg) ((_seg) >> ALLTOALLV_HYBRID_SEG_DIGIT)

#define SET_BRUCK_DIGIT(_seg, _digit) \
    ((_seg) = (((_digit) << ALLTOALLV_HYBRID_SEG_DIGIT) + ((_seg) & UCC_MASK(ALLTOALLV_HYBRID_SEG_DIGIT))))

static inline ucc_rank_t get_pairwise_send_peer(ucc_rank_t trank, ucc_rank_t tsize,
                                                ucc_rank_t step)
{
    return (trank + step) % tsize;
}

static inline ucc_rank_t get_pairwise_recv_peer(ucc_rank_t trank, ucc_rank_t tsize,
                                                ucc_rank_t step)
{
    return (trank - step + tsize) % tsize;
}

static void send_completion(void *request, ucs_status_t status,
                            void *user_data)
{
    ucc_tl_ucp_alltoallv_hybrid_merge_bin_t *bin =
        (ucc_tl_ucp_alltoallv_hybrid_merge_bin_t*)user_data;

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(bin->task), "failure in alltoallv_hybird completion %s",
                 ucs_status_string(status));
        bin->task->super.status = ucs_status_to_ucc_status(status);
    }

    if (request) {
        ucp_request_free(request);
    }

    ucc_atomic_add32(&bin->task->tagged.send_completed, 1);
    bin->len = 0;
}

static ucc_status_t ucc_tl_ucp_alltoallv_hybrid_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->alltoallv_hybrid.scratch_mc_header) {
        ucc_mc_free(task->alltoallv_hybrid.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

static inline void radix_setup(ucc_tl_ucp_task_t *task, int id, ucc_rank_t tsize,
                               size_t merge_buf_size,
                               ucc_tl_ucp_alltoallv_hybrid_buf_meta_t** meta,
                               void** merge_buf, void** tmp_buf)
{
    ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *bm = TASK_LB(task, tsize);

    (*meta)      = &bm[id];
    (*merge_buf) = TASK_BUF(task, id, tsize);
    (*tmp_buf)   = PTR_OFFSET(*merge_buf, merge_buf_size);
}

/* get the number of blocks of data to be sent for the given step and edge in the algorithm. */
static inline int get_send_block_count(ucc_rank_t tsize, int radix,
                                       int node_edge_id, int radix_pow)
{
    int k          = tsize / radix_pow;
    int send_count = (k / radix) * radix_pow;

    if ((k % radix) > node_edge_id) {
        send_count += radix_pow;
    } else if ((k % radix) == node_edge_id) {
        send_count += (tsize % radix_pow);
    }

    return send_count;
}

static inline int calculate_head_size(int snd_count, size_t dt_size)
{
    int head_num_elements = 0;

    if (dt_size <= sizeof(unsigned int)) {
        head_num_elements = ((snd_count + 1) * sizeof(unsigned int)) / dt_size;
    } else {
        /* The size of dtype is greater than unsigned int */
        if (((snd_count + 1) * sizeof(unsigned int)) % dt_size == 0) {
            /* The size of head can fit */
            head_num_elements = ((snd_count + 1) * sizeof(unsigned int)) / dt_size;
        } else {
            /* The size of head can not fit*/
            head_num_elements = ((snd_count + 1) * sizeof(unsigned int)) / dt_size + 1;
        }
    }
    return head_num_elements;
}

static int fit_in_send_buffer(int num,
                              ucc_tl_ucp_alltoallv_hybrid_buf_meta_t* meta,
                              int size_req, int mem_size)
{
    int *cur_bin  = &meta->cur_bin;
    int  start_ok = 1;
    int  bin_rdy  = -1;
    int i, k, pos, valid_pos;

    ucc_assert(size_req <= mem_size);
    for (i = 0; i < num; i++) {
        if (meta->bins[i].len == 0) {
            bin_rdy = i;
            break;
        }
    }
    *cur_bin = bin_rdy;

    if (bin_rdy == -1) {
        /* no free bins, exit */
        return -1;
    }

    for (i = 0; i < num; i++) {
        if ((meta->bins[i].len > 0) && (meta->bins[i].start < size_req)) {
            start_ok = 0;
            break;
        }
    }

    if (start_ok == 1) {
        /* we have enough empty space at the beggining of buffer */
        return 0;
    }

    /* at this point at least one bin is not empty */
    for (k = 0; k < num; k++) { /* complexity O(num_bins^2) */
        if (meta->bins[k].len > 0) {
            /* bin is not empty, check if there is enough space after it*/
            valid_pos = 1;
            pos = meta->bins[k].start + meta->bins[k].len;
            if (pos + size_req < mem_size) {
                /* there is enough space till the end of the buffer,
                 * need to check collisions with other bins
                 */
                for (i = 0; i < num; i++) {
                    if ((i == k) || (meta->bins[i].len == 0)) {
                        /* skip current bin and empty bins */
                        continue;
                    }
                    if (meta->bins[i].start < (pos + size_req) &&
                        pos < meta->bins[i].start + meta->bins[i].len) {
                        /* two regions overlap => collision */
                        valid_pos = 0;
                        break;
                    }
                }

                if (valid_pos) {
                    /* found a slot, no collisions with other bins */
                    return pos;
                }
            }
        }
    }
    /* no available slots */
    return -1;
}

static inline
ucc_status_t get_send_buffer(void *p_tmp_send_region,
                             ucc_tl_ucp_alltoallv_hybrid_buf_meta_t* meta,
                             int num_bins, int required_buf_size,
                             ucc_tl_ucp_task_t *task, void **buf)
{
    int merge_buf_size = task->alltoallv_hybrid.merge_buf_size;
    int polls          = 0;
    int send_buf_offset;

    /* get send buffer pointer */
    send_buf_offset = fit_in_send_buffer(num_bins, meta, required_buf_size,
                                         merge_buf_size);
    if (send_buf_offset < 0) {
        do {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            send_buf_offset = fit_in_send_buffer(num_bins, meta,
                                                 required_buf_size,
                                                 merge_buf_size);
        } while ((send_buf_offset < 0) && (polls++ < task->n_polls));
        if (send_buf_offset < 0) {
            return UCC_INPROGRESS;
        }
    }

    meta->bins[meta->cur_bin].start = send_buf_offset;
    *buf = PTR_OFFSET(p_tmp_send_region, send_buf_offset);
    return UCC_OK;
}

/* Pack data for sending
 * packed_send_buf (output) contains the send buffer that is ready to be sent
 * returns length of the buffer to be sent
 */
static size_t pack_send_data(ucc_tl_ucp_task_t *task, int step,
                             int send_count, int node_edge_id, size_t dt_size,
                             int send_limit, int step_header_size,
                             void *packed_send_buf)
{
    ucc_tl_ucp_team_t *team                 = TASK_TEAM(task);
    ucc_rank_t         tsize                = UCC_TL_TEAM_SIZE(team);
    int               *BytesForPacking      = TASK_TMP(task, tsize);
    char              *seg_st               = TASK_SEG(task, tsize);
    int                tmp_send_region_size = task->alltoallv_hybrid.merge_buf_size;
    uint32_t           radix                = task->alltoallv_hybrid.radix;
    void             **src_iovec            = TASK_PTRS(task, tsize);
    void              *user_sbuf            = TASK_ARGS(task).src.info_v.buffer;
    int                sparse_cnt           = 0;
    int                n                    = send_count;
    int                logi                 = 1;
    void              *op_metadata          = task->alltoallv_hybrid.scratch_mc_header->addr;
    ucc_rank_t index;
    int send_len, snd_offset, i, k;
    unsigned int temp_count;
    char loc;
    size_t offset;

    for (i = 0; i < n; i++) {
        BytesForPacking[i] = send_limit;
    }

    while (logi < n) {
        logi = logi * radix;
    }

    index = get_bruck_step_finish(tsize - 1, radix, node_edge_id, step);
    while (n > 0) {
        ucc_assert((index / step) % radix == node_edge_id);
        n--;
        if (logi > n) {
            logi = logi / radix;
        }
        temp_count = ((unsigned int *)op_metadata)[index + tsize];
        if (temp_count == COUNT_DIRECT) {
            /* data will be directly sent */
            send_len = 0;
        } else {
            loc        = GET_BRUCK_DIGIT(seg_st[index]);
            snd_offset = ((unsigned int *)op_metadata)[index] * dt_size;
            send_len   = dt_size * temp_count;
            if (loc) {
                /* data is in scratch buffer */
                src_iovec[n] = PTR_OFFSET(TASK_BUF(task, (loc - 1), tsize),
                                          tmp_send_region_size + snd_offset);
            } else {
                /* data is in user buffer */
                if ((send_len <= BytesForPacking[n]) &&
                    (send_len < MAX_BRUCK || n < step)) {
                    src_iovec[n] = PTR_OFFSET(user_sbuf, snd_offset);
                } else {
                    /* direct exchange */
                    temp_count = COUNT_DIRECT;
                    task->alltoallv_hybrid.num2send++;
                    /* mark the send as direct */
                    SET_DIRECT_SEND(seg_st[index]);
                    send_len = 0;
                }
            }
        }
        ((unsigned int *)packed_send_buf)[n + 1] = temp_count;
        if (send_len > 0) {
            sparse_cnt++;
        }

        if (logi > 0) {
            BytesForPacking[n % logi] += BytesForPacking[n] - send_len;
        }
        index = GET_PREV_BRUCK_NUM(index, radix, step);
    }
    /* if there is data for less than half the destinations, send meta data
     * in sparse format.
     */
    if ((sparse_cnt * 2) < send_count) {
        /* sparse format: store nonzero messages and source rank index */
        ((unsigned int *)packed_send_buf)[0] = sparse_cnt;
        /* compute space requirement for sparse data exchange header */
        offset = calculate_head_size(sparse_cnt * 2, dt_size) * dt_size;
        i = k = 0;
        while (k < sparse_cnt) {
            unsigned int cur_len = ((unsigned int *)packed_send_buf)[i + 1];
            if (cur_len != 0 && cur_len != COUNT_DIRECT) {
                BytesForPacking[2 * k]     = i;
                BytesForPacking[2 * k + 1] = cur_len;
                ++k;
            }
            ++i;
        }
        for (i = 0; i < sparse_cnt; i++){
            ((unsigned int *)packed_send_buf)[2 * i + 1] = BytesForPacking[2 * i];
            ((unsigned int *)packed_send_buf)[2 * i + 2] = BytesForPacking[2 * i + 1];
        }
        /* pack the send buffer */
        for (k = 0; k < sparse_cnt; ++k) {
            i          = ((unsigned int *)packed_send_buf)[2 * k + 1];
            temp_count = ((unsigned int *)packed_send_buf)[2 * k + 2];
            memcpy(PTR_OFFSET(packed_send_buf, offset), src_iovec[i],
                   temp_count * dt_size);
            offset = offset + temp_count * dt_size;
        }
    } else {
        /* dense format: store all message information even if message size is 0 */
        ((unsigned int *)packed_send_buf)[0] = DENSE_PACK_FORMAT;
        offset = step_header_size;
        for (i = 0; i < send_count; i++) {
            temp_count = ((unsigned int *)packed_send_buf)[i + 1];
            if (temp_count != COUNT_DIRECT) {
                memcpy(PTR_OFFSET(packed_send_buf, offset), src_iovec[i],
                       temp_count * dt_size);
                offset = offset + temp_count * dt_size;
            }
        }
    }

    return offset;
}

static inline
ucc_status_t send_data(void *buf, int send_size, ucc_rank_t dst,
                       ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *meta,
                       ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    ucc_assert(meta->bins[meta->cur_bin].len != 0);
    status = ucc_tl_ucp_send_cb(buf, send_size, UCC_MEMORY_TYPE_HOST, dst,
                                TASK_TEAM(task), task, send_completion,
                                (void*)&meta->bins[meta->cur_bin]);
    task->alltoallv_hybrid.phase = UCC_ALLTOALLV_HYBRID_PHASE_SENT;
    return status;
}

static
int receive_buffer_recycler(ucc_rank_t tsize, unsigned int* rcv_start, int* rcv_len,
                            char* seg_st, void* buf, size_t dt_size, int* tmp_buf,
                            int step, void* rbuf, int* rdisps, int my_group_index,
                            int radix, int node_edge_id)
{
    int cur   = 0;
    int mstep = step / radix;
    int i, k, in_buf, offset, idx;

    for (i = 0; i < tsize; ++i) {
        if (GET_BRUCK_DIGIT(seg_st[i]) == node_edge_id) {
            tmp_buf[i + tsize] = 1;
            ++cur;
        } else {
            tmp_buf[i + tsize] = 0;
        }
    }

    in_buf = cur;
    --cur;
    while (cur >= 0) {
        for (i = tsize - 1; i >= 0; --i) {
            if (tmp_buf[i + tsize] && ((i / mstep) % radix == node_edge_id)) {
                tmp_buf[cur]       = i;
                tmp_buf[i + tsize] = 0;
                --cur;
            }
        }
        mstep = mstep / radix;
    }
    k = 0;
    offset = 0;
    while (k < in_buf) {
        cur = tmp_buf[k];
        if ((rcv_start[cur] == COUNT_DIRECT) || (rcv_len[cur] == 0)) {
            ++k;
            continue;
        }
        if (cur < step) {
            /* final destination of data */
            idx = (my_group_index - cur + tsize) % tsize;
            memcpy(PTR_OFFSET(rbuf, rdisps[idx] * dt_size),
                   PTR_OFFSET(buf, rcv_start[cur] * dt_size),
                   rcv_len[cur] * dt_size);
            rcv_start[cur] = COUNT_DIRECT;
            rcv_len[cur] = 0;
            seg_st[cur] = seg_st[cur] % 4;
        } else {
            if (offset < rcv_start[cur]) {
                memmove(PTR_OFFSET(buf, offset * dt_size),
                        PTR_OFFSET(buf, rcv_start[cur] * dt_size),
                        rcv_len[cur] * dt_size);
                rcv_start[cur] = offset;
                offset += rcv_len[cur];
            } else {
                ucc_assert(offset == rcv_start[cur]);
                offset += rcv_len[cur];
            }
        }
        ++k;
    }

    return offset;
}

static
ucc_status_t post_recv(ucc_rank_t recvfrom, ucc_rank_t tsize, size_t dt_size,
                       int node_edge_id, int step, ucc_rank_t trank,
                       void *op_metadata, int tmp_buf_size, char *seg_st,
                       ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *meta,
                       int step_buf_size, int *BytesForPacking,
                       void *p_tmp_recv_region, ucc_tl_ucp_task_t *task)
{
    int          *rdisps    = (int*)TASK_ARGS(task).dst.info_v.displacements;
    void         *user_rbuf = TASK_ARGS(task).dst.info_v.buffer;
    uint32_t      radix     = task->alltoallv_hybrid.radix;
    ucc_status_t  status    = UCC_OK;
    int new_offset;
    void* dst_buf;

    if (task->alltoallv_hybrid.phase != UCC_ALLTOALLV_HYBRID_PHASE_SENT) {
        return UCC_OK;
    }

    /* check if we have space for maximum receive. If not, recycle */
    if (meta->offset * dt_size + step_buf_size > tmp_buf_size) {
        new_offset = receive_buffer_recycler(tsize, (unsigned int *)op_metadata,
                                             (int *)op_metadata + tsize,
                                             seg_st, p_tmp_recv_region, dt_size,
                                             BytesForPacking, step, user_rbuf,
                                             rdisps, trank, radix, node_edge_id);
        meta->offset = new_offset;
    }
    ucc_assert(meta->offset * dt_size + step_buf_size <= tmp_buf_size);
    dst_buf = PTR_OFFSET(p_tmp_recv_region, meta->offset * dt_size);
    status = ucc_tl_ucp_recv_nb(dst_buf, step_buf_size, UCC_MEMORY_TYPE_HOST,
                                recvfrom, TASK_TEAM(task), task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->alltoallv_hybrid.phase = UCC_ALLTOALLV_HYBRID_PHASE_START;
    task->alltoallv_hybrid.cur_radix++;
    if (task->alltoallv_hybrid.cur_radix == radix) {
        /* setup the control variables for the next phase of processing */
        task->alltoallv_hybrid.phase     = UCC_ALLTOALLV_HYBRID_PHASE_RECV;
        task->alltoallv_hybrid.cur_radix = 1;
    }

    return UCC_OK;
}

/* complete all receives in current step */
static ucc_status_t complete_current_step_receives(ucc_rank_t tsize, int step,
                                                   size_t dt_size, ucc_rank_t trank,
                                                   void *op_metadata, char *seg_st,
                                                   ucc_tl_ucp_task_t *task)
{
    size_t    tmp_send_region_size = task->alltoallv_hybrid.merge_buf_size;
    uint32_t  radix                = task->alltoallv_hybrid.radix;
    int      *rcounts              = (int*)TASK_ARGS(task).dst.info_v.counts;
    int       polls                = 0;
    ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *meta;
    unsigned int rcv_sparse, next_p, cur_p;
    void *p_tmp_recv_region, *p_tmp_send_region, *dst_buf, *temp_offset;
    int i, k, n, node_edge_id, recv_count, cur_buf_length, recv_size;

    while ((task->tagged.recv_posted != task->tagged.recv_completed) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
    }

    if (task->tagged.recv_posted != task->tagged.recv_completed) {
        return UCC_INPROGRESS;
    }

    while (task->alltoallv_hybrid.cur_radix < radix) {
        node_edge_id = task->alltoallv_hybrid.cur_radix;
        radix_setup(task, node_edge_id - 1, tsize, tmp_send_region_size,
                    &meta, &p_tmp_send_region, &p_tmp_recv_region);

        n = get_send_block_count(tsize, radix, node_edge_id, step);
        if (!n) {
            /* nothing to be done here */
            task->alltoallv_hybrid.cur_radix++;
            continue;
        }

        /* fill in the information to be be used in subsequenst aggrgation
         * steps to extract data received for packing.
         */
        dst_buf = PTR_OFFSET(p_tmp_recv_region, meta->offset * dt_size);
        rcv_sparse = ((unsigned int *)dst_buf)[0];
        i = get_bruck_step_start(step, node_edge_id);
        if (rcv_sparse == DENSE_PACK_FORMAT) {
            recv_count = 0;
            recv_size = calculate_head_size(n, dt_size);
            temp_offset = PTR_OFFSET(dst_buf, recv_size * dt_size);
            /* this is where we parse the received bruck-like packet and
             * set the pointers to point on the important data segments.
             */
            while (i < tsize) {
                cur_buf_length = ((unsigned int *)dst_buf)[1 + recv_count];
                if (cur_buf_length != COUNT_DIRECT) {
                    ((unsigned int *)op_metadata)[i] =
                        ((char *)temp_offset-(char *)p_tmp_recv_region) / dt_size;
                    /* mark data received destined to rank i, and its length */
                    SET_BRUCK_DIGIT(seg_st[i], node_edge_id);
                    ((int *)op_metadata)[i + tsize] = cur_buf_length;
                    recv_size += cur_buf_length;
                    temp_offset = PTR_OFFSET(temp_offset, cur_buf_length * dt_size);
                } else {
                    /* data will be sent pairwise */
                    ((int *)op_metadata)[i]         = (int)COUNT_DIRECT;
                    ((int *)op_metadata)[i + tsize] = (int)COUNT_DIRECT;
                    if (i < (step * radix)) {
                        int pairwise_src = (trank - i + tsize) % tsize;
                        if (rcounts[pairwise_src] > 0) {
                            task->alltoallv_hybrid.num2recv++;
                            SET_DIRECT_RECV(seg_st[i]);
                        }
                    }
                }
                ++recv_count;
                i = GET_NEXT_BRUCK_NUM(i, radix, step);
            }
        } else {
            recv_size = calculate_head_size(2*rcv_sparse, dt_size);
            temp_offset= PTR_OFFSET(dst_buf, recv_size*dt_size);
            k = 0;
            if (rcv_sparse > 0) {
                next_p = ((unsigned int *)dst_buf)[2 * k + 1];
            } else {
                next_p = tsize;
            }

            cur_p = 0;
            while (i < tsize) {
                if (cur_p == next_p) {
                    cur_buf_length = ((unsigned int *)dst_buf)[2 * k + 2];
                    ((int *)op_metadata)[i] =
                        (((char *)temp_offset - (char *)p_tmp_recv_region) / dt_size);
                    ((int *)op_metadata)[i + tsize] = cur_buf_length;

                    recv_size+= cur_buf_length;
                    temp_offset = PTR_OFFSET(temp_offset, cur_buf_length * dt_size);
                    SET_BRUCK_DIGIT(seg_st[i], node_edge_id);
                    ++k;
                    if (k < rcv_sparse) {
                        next_p= ((unsigned int *)dst_buf)[2*k + 1];
                    } else {
                        next_p = tsize;
                    }
                } else {
                    ((int *)op_metadata)[i]         = (int)COUNT_DIRECT;
                    ((int *)op_metadata)[i + tsize] = (int)COUNT_DIRECT;
                    if (i < (step * radix)) {
                        int pairwise_src = (trank - i + tsize) % tsize;
                        if (rcounts[pairwise_src] > 0) {
                            task->alltoallv_hybrid.num2recv++;
                            SET_DIRECT_RECV(seg_st[i]);
                        }
                    }
                }
                ++cur_p;
                i = GET_NEXT_BRUCK_NUM(i, radix, step);
            }
            ucc_assert(next_p == tsize);
        }
        meta->offset += recv_size;
        task->alltoallv_hybrid.cur_radix++;
    }

    return UCC_OK;
}

static inline void hybrid_reverse_rotation(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team           = TASK_TEAM(task);
    ucc_rank_t         tsize          = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         trank          = UCC_TL_TEAM_RANK(team);
    char              *seg_st         = TASK_SEG(task, tsize);
    size_t             merge_buf_size = task->alltoallv_hybrid.merge_buf_size;
    void              *user_sbuf      = TASK_ARGS(task).src.info_v.buffer;
    void              *user_rbuf      = TASK_ARGS(task).dst.info_v.buffer;
    int               *rdisps         = (int*)TASK_ARGS(task).dst.info_v.displacements;
    void              *metainfo       = task->alltoallv_hybrid.scratch_mc_header->addr;
    size_t dt_size;
    int i, idx, cur_buf_index, cur_buf_size;
    char loc;
    void *lb;

    dt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);
    for (i = 0; i < tsize; i++) {
        cur_buf_index = ((int *)metainfo)[i];
        cur_buf_size  = ((int *)metainfo)[i + tsize];
        if (cur_buf_index != COUNT_DIRECT ) {
            loc = GET_BRUCK_DIGIT(seg_st[i]);
            idx = (trank - i + tsize) % tsize;
            if (loc == 0) {
                /* This block of data is in user send buffer */
                memcpy(PTR_OFFSET(user_rbuf, rdisps[idx] * dt_size),
                       PTR_OFFSET(user_sbuf, cur_buf_index * dt_size),
                       cur_buf_size * dt_size);
            } else {
                /* This block of data is in scratch buffer */
                lb = TASK_BUF(task, loc - 1, tsize);
                memcpy(PTR_OFFSET(user_rbuf, rdisps[idx] * dt_size),
                       PTR_OFFSET(lb, merge_buf_size + cur_buf_index * dt_size),
                       cur_buf_size * dt_size);
            }
        }
    }
}

static
ucc_status_t pairwise_manager(ucc_rank_t trank, ucc_rank_t tsize,
                              size_t dt_size, ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team             = TASK_TEAM(task);
    char              *seg_st           = TASK_SEG(task, tsize);
    void              *user_sbuf        = TASK_ARGS(task).src.info_v.buffer;
    void              *user_rbuf        = TASK_ARGS(task).dst.info_v.buffer;
    int               *s_disps          = (int*)TASK_ARGS(task).src.info_v.displacements;
    int               *r_disps          = (int*)TASK_ARGS(task).dst.info_v.displacements;
    int               *scounts          = (int*)TASK_ARGS(task).src.info_v.counts;
    int               *rcounts          = (int*)TASK_ARGS(task).dst.info_v.counts;
    ucc_rank_t        *cur              = &task->alltoallv_hybrid.cur_out;
    int                chunk_num_limit  = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_hybrid_pairwise_num_posts;
    int                chunk_byte_limit = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_hybrid_chunk_byte_limit;
    ucc_status_t status;
    void* mem_dst;
    int pairwise_dest, msg_size;

    if ((task->alltoallv_hybrid.num_in < chunk_num_limit) &&
        (task->alltoallv_hybrid.traffic_in <= chunk_byte_limit) &&
        (task->alltoallv_hybrid.traffic_out <= chunk_byte_limit)) {
        if ((task->alltoallv_hybrid.num2send == 0) &&
            (task->alltoallv_hybrid.num2recv == 0)) {
            return UCC_OK;
        }

        while (!(IS_DIRECT_SEND(seg_st[*cur]) || IS_DIRECT_RECV(seg_st[*cur]))) {
            ++(*cur);
            ucc_assert(*cur < tsize);
        }

        if (IS_DIRECT_SEND(seg_st[*cur])) {
            pairwise_dest = get_pairwise_send_peer(trank, tsize, *cur);
            mem_dst = PTR_OFFSET(user_sbuf, s_disps[pairwise_dest] * dt_size);
            msg_size = scounts[pairwise_dest] * dt_size;
            task->alltoallv_hybrid.traffic_out += msg_size;
            if ((task->alltoallv_hybrid.num_in > 0) &&
                (task->alltoallv_hybrid.traffic_out > chunk_byte_limit)) {
                /* too much outgoing trafic, exit */
                return UCC_OK;
            }

            status = ucc_tl_ucp_send_nb(mem_dst, msg_size, UCC_MEMORY_TYPE_HOST,
                                        pairwise_dest, team, task);
            if (ucc_unlikely(UCC_OK != status)) {
                return status;
            }
            seg_st[(*cur)] = seg_st[(*cur)] - 1;
            task->alltoallv_hybrid.num2send--;
        }

        if (IS_DIRECT_RECV(seg_st[*cur])) {
            pairwise_dest = get_pairwise_recv_peer(trank, tsize, *cur);
            mem_dst = PTR_OFFSET(user_rbuf, r_disps[pairwise_dest] * dt_size);
            msg_size = rcounts[pairwise_dest] * dt_size;
            status = ucc_tl_ucp_recv_nb(mem_dst, msg_size, UCC_MEMORY_TYPE_HOST,
                                        pairwise_dest, team, task);
            if (ucc_unlikely(UCC_OK != status)) {
                return status;
            }
            task->alltoallv_hybrid.traffic_in += msg_size;
            seg_st[(*cur)] = seg_st[(*cur)] - 2;
            task->alltoallv_hybrid.num2recv--;
        }
        task->alltoallv_hybrid.num_in++;
        ++(*cur);
    } else {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return UCC_INPROGRESS;
        }
        task->alltoallv_hybrid.num_in      = 0;
        task->alltoallv_hybrid.traffic_in  = 0;
        task->alltoallv_hybrid.traffic_out = 0;
    }

    return UCC_OK;
}

static void ucc_tl_ucp_alltoallv_hybrid_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task            = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team            = TASK_TEAM(task);
    ucc_rank_t         gsize           = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         grank           = UCC_TL_TEAM_RANK(team);
    uint32_t           radix           = task->alltoallv_hybrid.radix;
    char              *seg_st          = TASK_SEG(task, gsize);
    void              *op_metadata     = task->alltoallv_hybrid.scratch_mc_header->addr;
    size_t             merge_buf_size  = task->alltoallv_hybrid.merge_buf_size;
    size_t             send_limit      = task->alltoallv_hybrid.byte_send_limit;
    int               *BytesForPacking = TASK_TMP(task, gsize);
    size_t             buff_size       = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_hybrid_buff_size;
    size_t             tmp_buf_size    = buff_size - (merge_buf_size + sizeof(ucc_tl_ucp_alltoallv_hybrid_buf_meta_t));
    size_t             dt_size         = ucc_dt_size(coll_task->bargs.args.dst.info_v.datatype);
    ucc_rank_t sendto, recvfrom;
    int node_edge_id, step, i, istep, step_header_size, step_buf_size;
    size_t snd_count, send_size;
    void *p_tmp_recv_region, *p_tmp_send_region, *buf;
    ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *meta;
    ucc_status_t status;

    istep = 1;
    for (i = 1; i < task->alltoallv_hybrid.iteration; i++) {
        istep *= radix;
    }

    /* Loop over algorithm stage - number of stages is ceil(log_k(N)),
     * where k is algorithm radix and N is communicator size
     */
    for (step = istep; step < gsize; step *= radix) {
        while ((task->alltoallv_hybrid.phase == UCC_ALLTOALLV_HYBRID_PHASE_START) ||
               (task->alltoallv_hybrid.phase == UCC_ALLTOALLV_HYBRID_PHASE_SENT)) {
            /* initiate sends
             * set the current substage index. node_edge_id=1, 2, ..., k-1
             */
            node_edge_id = task->alltoallv_hybrid.cur_radix;
            radix_setup(task, node_edge_id - 1, gsize, merge_buf_size,
                        &meta, &p_tmp_send_region, &p_tmp_recv_region);
            /* figure out number of desinaiton ranks the will be included in the current send */
            snd_count = get_send_block_count(gsize, radix, node_edge_id, step);
            if (!snd_count) {
                task->alltoallv_hybrid.cur_radix++;
                if (task->alltoallv_hybrid.cur_radix == radix) {
                    task->alltoallv_hybrid.phase     = UCC_ALLTOALLV_HYBRID_PHASE_RECV;
                    task->alltoallv_hybrid.cur_radix = 1;
                }
                continue;
            }

            /* peers to communicate with in the rth exchange of the current stage */
            sendto   = get_bruck_send_peer(grank, gsize, step, node_edge_id);
            recvfrom = get_bruck_recv_peer(grank, gsize, step, node_edge_id);

            /* Send aggregated data */
            step_header_size = calculate_head_size(snd_count, dt_size) * dt_size;
            step_buf_size    = (send_limit * snd_count) + step_header_size;
            if (task->alltoallv_hybrid.phase == UCC_ALLTOALLV_HYBRID_PHASE_START) {
                /* Initialize the space data destined to process i,
                 * where i is the index in the array. Buddy buffer algorithm may
                 * increase this value.
                 */
                status = get_send_buffer(p_tmp_send_region, meta, NUM_BINS,
                                         step_buf_size, task, &buf);
                if (UCC_OK != status) {
                    task->super.status = status;
                    goto out;
                }

                /* pack the data */
                send_size = pack_send_data(task, step, snd_count, node_edge_id,
                                           dt_size, send_limit,
                                           step_header_size, buf);
                ucc_assert(step_buf_size >= send_size);
                meta->bins[meta->cur_bin].len = send_size;

                status = send_data(buf, send_size, sendto, meta, task);
                if (ucc_unlikely(UCC_OK != status)) {
                    task->super.status = status;
                    goto out;
                }
            }

            status = post_recv(recvfrom, gsize, dt_size, node_edge_id,
                               step, grank, op_metadata, tmp_buf_size, seg_st,
                               meta, step_buf_size, BytesForPacking,
                               p_tmp_recv_region, task);
            if (ucc_unlikely(UCC_OK != status)) {
                task->super.status = status;
                goto out;
            }
        }

        status = complete_current_step_receives(gsize, step, dt_size, grank,
                                                op_metadata, seg_st, task);
        if (UCC_OK != status) {
            task->super.status = status;
            goto out;
        }
        task->alltoallv_hybrid.phase     = UCC_ALLTOALLV_HYBRID_PHASE_START;
        task->alltoallv_hybrid.cur_radix = 1;
        task->alltoallv_hybrid.iteration++;
    }
    /* The brucks iterations are done. Now we send and recv all the
     * pairwise we didn't already send and receive
     */
    while ((task->alltoallv_hybrid.num2recv > 0) ||
           (task->alltoallv_hybrid.num2send > 0)) {
        status = pairwise_manager(grank, gsize, dt_size, task);
        if (UCC_OK != status) {
            task->super.status = status;
            goto out;
        }
    }

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    hybrid_reverse_rotation(task);
    task->super.status = UCC_OK;
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoallv_hybrid_done", 0);
    }
}

static inline void meta_init(ucc_tl_ucp_alltoallv_hybrid_buf_meta_t* meta,
                             ucc_tl_ucp_task_t *task)
{
    int i;

    meta->cur_bin = 0;
    meta->offset  = 0;
    for (i = 0; i < NUM_BINS; i++) {
        meta->bins[i].len  = 0;
        meta->bins[i].task = task;
    }
}

static inline void copy_brucks_rotation(void *scratch,
                                        void *scounts, void *sdisps,
                                        ucc_rank_t trank, ucc_rank_t tsize)
{

    size_t size = sizeof(int); /* currently support for 32 bit counts */

    memcpy(scratch, PTR_OFFSET(sdisps, trank * size), (tsize - trank) * size);
    memcpy(PTR_OFFSET(scratch, tsize * size), PTR_OFFSET(scounts, trank * size),
           (tsize - trank) * size);
    /* Copy the rest part */
    if (trank != 0) {
        memcpy(PTR_OFFSET(scratch, (tsize - trank) * size), sdisps, trank * size);
        memcpy(PTR_OFFSET(scratch, (tsize - trank + tsize) * size), scounts,
               trank * size);
    }
}

static ucc_status_t ucc_tl_ucp_alltoallv_hybrid_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    uint32_t           radix  = task->alltoallv_hybrid.radix;
    ucc_rank_t         tsize  = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         trank  = UCC_TL_TEAM_RANK(team);
    ucc_coll_args_t   *args   = &TASK_ARGS(task);
    ucc_tl_ucp_alltoallv_hybrid_buf_meta_t *lbm;
    int i;

    lbm = TASK_LB(task, tsize);
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoallv_hybrid_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    copy_brucks_rotation(task->alltoallv_hybrid.scratch_mc_header->addr,
                         args->src.info_v.counts, args->src.info_v.displacements,
                         trank, tsize);

    task->alltoallv_hybrid.num_in      = 0;
    task->alltoallv_hybrid.cur_out     = 1;
    task->alltoallv_hybrid.cur_radix   = 1;
    task->alltoallv_hybrid.traffic_in  = 0;
    task->alltoallv_hybrid.traffic_out = 0;
    task->alltoallv_hybrid.num2send    = 0;
    task->alltoallv_hybrid.num2recv    = 0;
    task->alltoallv_hybrid.phase       = UCC_ALLTOALLV_HYBRID_PHASE_START;
    task->alltoallv_hybrid.iteration   = 1;

    memset(TASK_SEG(task, tsize), 0, tsize * sizeof(char));
    for (i = 0; i < radix - 1; ++i) {
        meta_init(&lbm[i], task);
    }
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoallv_hybrid_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team    = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         tsize      = UCC_TL_TEAM_SIZE(tl_team);
    uint32_t           radix      = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoallv_hybrid_radix;
    size_t             buff_size  = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoallv_hybrid_buff_size;
    uint32_t           snd_size   = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoallv_hybrid_num_scratch_sends;
    uint32_t           rcv_size   = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoallv_hybrid_num_scratch_recvs;
    ucc_tl_ucp_task_t *task;
    size_t             scratch_size, calc_limit, max_snd_count, dt_size;
    ucc_status_t       status;

    if (UCC_COLL_ARGS_DISPL64(&coll_args->args) ||
        UCC_COLL_ARGS_COUNT64(&coll_args->args) ||
        coll_args->args.src.info_v.mem_type != UCC_MEMORY_TYPE_HOST ||
        coll_args->args.dst.info_v.mem_type != UCC_MEMORY_TYPE_HOST) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }
    task->super.post     = ucc_tl_ucp_alltoallv_hybrid_start;
    task->super.progress = ucc_tl_ucp_alltoallv_hybrid_progress;
    task->super.finalize = ucc_tl_ucp_alltoallv_hybrid_finalize;

    dt_size = ucc_dt_size(coll_args->args.dst.info_v.datatype);

    task->alltoallv_hybrid.radix = radix;
    scratch_size = 2 * tsize * sizeof(int) /* rotated scounts and sdispls */
        + 2 * tsize * sizeof(int) /* BytesForPacking */
        + tsize * sizeof(char*) /* seg_st */
        + (radix - 1) * sizeof(ucc_tl_ucp_alltoallv_hybrid_buf_meta_t)
        + ucc_div_round_up(tsize, 2) * sizeof(void*)
        + (radix - 1) * buff_size;

    status = ucc_mc_alloc(&task->alltoallv_hybrid.scratch_mc_header,
                          scratch_size, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }

    /* TODO: fix for radix > 2 */
    max_snd_count = ucc_ceil(tsize, radix) / radix;

    calc_limit = ((buff_size - 256) / (snd_size + rcv_size) -
                  ucc_ceil(sizeof(int) * (max_snd_count + 1), dt_size)) / max_snd_count;
    calc_limit -= (calc_limit % 4);
    ucc_assert(calc_limit > 0);
    task->alltoallv_hybrid.byte_send_limit = calc_limit;
    task->alltoallv_hybrid.merge_buf_size  =
        ALIGN((calc_limit*max_snd_count + ucc_ceil(sizeof(int) * (max_snd_count + 1), dt_size)) * snd_size);

    *task_h = &task->super;
    return UCC_OK;
}
