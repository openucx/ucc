/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_LIST_H_
#define UCC_LIST_H_

#include "config.h"
#include <ucs/datastruct/list.h>

#define ucc_list_link_t        ucs_list_link_t
#define ucc_list_head_init     ucs_list_head_init
#define ucc_list_add_tail      ucs_list_add_tail
#define ucc_list_del           ucs_list_del
#define ucc_list_for_each_safe ucs_list_for_each_safe
#endif
