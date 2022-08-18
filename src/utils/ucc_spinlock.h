/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SPINLOCK_H_
#define UCC_SPINLOCK_H_

#include "config.h"
#include "ucs/type/spinlock.h"

#define ucc_spinlock_t       ucs_spinlock_t
#define ucc_spinlock_init    ucs_spinlock_init
#define ucc_spinlock_destroy ucs_spinlock_destroy
#define ucc_spin_lock        ucs_spin_lock
#define ucc_spin_try_lock    ucs_spin_try_lock
#define ucc_spin_unlock      ucs_spin_unlock

#define ucc_recursive_spinlock_t        ucs_recursive_spinlock_t
#define ucc_recursive_spinlock_init     ucs_recursive_spinlock_init
#define ucc_recursive_spinlock_destroy  ucs_recursive_spinlock_destroy
#define ucc_recursive_spin_is_owner     ucs_recursive_spin_is_owner
#define ucc_recursive_spin_lock         ucs_recursive_spin_lock
#define ucc_recursive_spin_trylock      ucs_recursive_spin_trylock
#define ucc_recursive_spin_unlock       ucs_recursive_spin_unlock

#endif
