/**
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_DEBUG_H_
#define UCC_DEBUG_H_

#include "config.h"
#include "ucc/api/ucc.h"

static inline void ucc_check_wait_for_debugger(ucc_rank_t ctx_rank)
{
    const char *wait_for_rank;
    if (NULL != (wait_for_rank = getenv("UCC_DEBUGGER_WAIT"))) {
	volatile int waiting = 1;
	if (atoi(wait_for_rank) == ctx_rank) {
	    char hostname[256];
	    gethostname(hostname, sizeof(hostname));
	    printf("PID %d (ctx rank %d) waiting for attach on %s\n"
		   "Set var waiting = 0 in debugger to continue\n",
		   getpid(), ctx_rank, hostname);
	    while (waiting) {
		sleep(1);
	    }
	}
    }
}

#endif

