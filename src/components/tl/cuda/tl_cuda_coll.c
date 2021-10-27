#include "tl_cuda_coll.h"
#include "alltoall/alltoall.h"

ucc_status_t ucc_tl_cuda_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
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
