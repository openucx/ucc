/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_TIME_H_
#define UCC_TIME_H_
#include "config.h"
#include <sys/time.h>

#define UCC_USEC_PER_SEC   1000000ul     /* Micro */

/**
 * @return The current accurate time, in seconds.
 */
static inline double ucc_get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / (double)UCC_USEC_PER_SEC);
}

#endif
