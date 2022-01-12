#include "tl_cuda_coll.h"
#include "alltoall/alltoall.h"
#include "utils/arch/cpu.h"

ucc_status_t ucc_tl_cuda_mem_info_get(void *ptr, size_t length,
                                      ucc_tl_cuda_team_t *team,
                                      ucc_tl_cuda_mem_info_t *mi)
{
    ucc_mem_attr_t mem_attr;
    ucc_status_t   status;

    mem_attr.field_mask   = UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                            UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
    mem_attr.alloc_length = length;
    status = ucc_mc_get_mem_attr(ptr, &mem_attr);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    mi->ptr    = mem_attr.base_address;
    mi->length = mem_attr.alloc_length;
    mi->offset = (ptrdiff_t)ptr - (ptrdiff_t)mi->ptr;
    CUDACHECK_GOTO(cudaIpcGetMemHandle(&mi->handle, mi->ptr), exit, status,
                   UCC_TL_TEAM_LIB(team));
exit:
    return status;
}

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_cuda_alltoall_init(coll_args, team, task_h);
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
}

ucc_status_t ucc_tl_cuda_shm_barrier_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier)
{
    if (rank == 0) {
        barrier->size = size;
        barrier->count = 0;
        barrier->sense = 0;
    }
    barrier->state[rank]       = UCC_OK;
    barrier->local_sense[rank] = 1;
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_shm_barrier_start(ucc_rank_t rank,
                                           ucc_tl_cuda_shm_barrier_t *barrier)
{
    ucc_rank_t pos = ucc_atomic_fadd32(&barrier->count, 1);

    barrier->state[rank] = UCC_INPROGRESS;
    if (pos == barrier->size - 1) {
        barrier->count = 0;
        ucc_memory_bus_fence();
        barrier->sense = barrier->local_sense[rank];
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_shm_barrier_test(ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier)
{
    if (barrier->sense != barrier->local_sense[rank]) {
        return UCC_INPROGRESS;
    }
    barrier->state[rank]       = UCC_OK;
    barrier->local_sense[rank] = 1 - barrier->local_sense[rank];
    return UCC_OK;
}
