/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_SPINLOCK_H_
#define UCC_SPINLOCK_H_

#include "config.h"
#include <ucs/type/spinlock.h>

#define ucc_spinlock_t         ucs_spinlock_t
#define ucc_spinlock_init      ucs_spinlock_init
#define ucc_spinlock_destroy   ucs_spinlock_destroy
#define ucc_spin_lock          ucs_spin_lock
#define ucc_spin_unlock        ucs_spin_unlock
#endif
