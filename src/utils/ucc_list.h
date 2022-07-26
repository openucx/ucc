/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define ucc_list_for_each      ucs_list_for_each
#define ucc_list_is_empty      ucs_list_is_empty
#define ucc_list_extract_head  ucs_list_extract_head
#define ucc_list_length        ucs_list_length
#define ucc_list_head          ucs_list_head
#define ucc_list_next          ucs_list_next
#define ucc_list_insert_after  ucs_list_insert_after
#define ucc_list_insert_before ucs_list_insert_before

#define ucc_list_destruct(_list, _elem_type, _elem_destruct, _member)          \
    do {                                                                       \
        _elem_type *_elem, *_tmp;                                              \
        ucc_list_for_each_safe(_elem, _tmp, _list, _member)                    \
        {                                                                      \
            ucc_list_del(&_elem->_member);                                     \
            _elem_destruct(_elem);                                             \
        }                                                                      \
    } while (0)

#endif
