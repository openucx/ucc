/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MALLOC_H_
#define UCC_MALLOC_H_

#include "config.h"
#include <stdlib.h>

#define ucc_malloc(_s, ...) malloc(_s)
#define ucc_calloc(_n, _s, ...) calloc(_n, _s)
#define ucc_realloc(_p, _s, ...) realloc(_p, _s)
#define ucc_free(_p) free(_p)

#endif
