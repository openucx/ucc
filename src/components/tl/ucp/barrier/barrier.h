#ifndef BARRIER_H_
#define BARRIER_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

ucc_status_t ucc_tl_ucp_barrier_init(ucc_tl_ucp_task_t *task);

#endif
