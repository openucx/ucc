/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_MALLOC_H_
#define UCC_MALLOC_H_

#include "config.h"
#include <stdlib.h>

#define ucc_malloc(_s, ...) malloc(_s)
#define ucc_posix_memalign(_ptr, _align, _size, ...) posix_memalign(_ptr, _align, _size)
#define ucc_calloc(_n, _s, ...) calloc(_n, _s)
#define ucc_realloc(_p, _s, ...) realloc(_p, _s)
#define ucc_free(_p) free(_p)

#endif
