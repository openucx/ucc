/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl_coll.h"


ucc_status_t ucc_nccl_collective_progress(ucc_coll_task_t *coll_task)
{
    cudaError_t cuda_st;
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);

    cuda_st = cudaEventQuery(task->completed);
    switch(cuda_st) {
    case cudaSuccess:
        coll_task->super.status = UCC_OK;
        return UCC_OK;
    case cudaErrorNotReady:
        return UCC_INPROGRESS;
    default:
        return UCC_ERR_NO_MESSAGE;
    }
}
ucc_status_t ucc_nccl_alltoall_start(ucc_coll_task_t *coll_task)
{
    // ptrdiff_t    sbuf, rbuf;
    // size_t       data_size;
    // int          group_size;
    // int          peer;
    // cudaStream_t *stream;

    // stream = (cudaStream_t*)args->stream.stream;
    // NCCLCHECK(ncclCommCount(req->team->nccl_comm, &group_size));
    // sbuf      = (ptrdiff_t)args->buffer_info.src_buffer;
    // rbuf      = (ptrdiff_t)args->buffer_info.dst_buffer;
    // data_size = args->buffer_info.len;

    // NCCLCHECK(ncclGroupStart());
    // for (peer = 0; peer < group_size; peer++) {
    //     NCCLCHECK(ncclSend((void*)(sbuf + peer*data_size),
    //                        data_size, ncclChar, peer,
    //                        req->team->nccl_comm,
    //                        *stream));
    //     NCCLCHECK(ncclRecv((void*)(rbuf + peer*data_size),
    //                        data_size, ncclChar, peer,
    //                        req->team->nccl_comm,
    //                        *stream));

    // }
    // NCCLCHECK(ncclGroupEnd());

    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_tl_nccl_alltoall_init(ucc_tl_nccl_task_t *task)
{
    task->super.super.status = UCC_INPROGRESS;
    task->super.post = ucc_nccl_alltoall_start;
    task->super.progress = ucc_nccl_collective_progress;
    return UCC_OK;
}
