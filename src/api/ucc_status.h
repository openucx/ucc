/**
 * @file ucc_status.h
 * @date 2020
 * @copyright Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_STATUS_H_
#define UCC_STATUS_H_


/**
 * @brief Status codes for the UCC operations
 */

typedef enum {
    /* Operation completed successfully */
    UCC_OK                              =   0,

    /* Operation is posted and is in progress */
    UCC_INPROGRESS                      =   1,

    /* Operation initialized but not posted */
    UCC_OPERATION_INITIALIZED           =   2,
    UCC_ERR_OP_NOT_SUPPORTED            =   3,
    UCC_ERR_NOT_IMPLEMENTED             =   4,
    UCC_ERR_INVALID_PARAM               =   5,
    UCC_ERR_NO_MEMORY                   =   6,
    UCC_ERR_NO_RESOURCE                 =   7,

    UCC_ERR_LAST                        = -100,
} ucc_status_t;

#endif
