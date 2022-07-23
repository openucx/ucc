/**
 * @file ucc_status.h
 * @date 2020
 * @copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_STATUS_H_
#define UCC_STATUS_H_

#ifdef __cplusplus
# define BEGIN_C_DECLS  extern "C" {
# define END_C_DECLS    }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

BEGIN_C_DECLS
/**
 * @ingroup UCC_UTILS
 * @brief Status codes for the UCC operations
 */

typedef enum {
    /* Operation completed successfully */
    UCC_OK                              =    0,

    UCC_INPROGRESS                      =    1, /*!< Operation is posted and is in progress */

    UCC_OPERATION_INITIALIZED           =    2, /*!< Operation initialized but not posted */

    /* Error status codes */
    UCC_ERR_NOT_SUPPORTED               =   -1,
    UCC_ERR_NOT_IMPLEMENTED             =   -2,
    UCC_ERR_INVALID_PARAM               =   -3,
    UCC_ERR_NO_MEMORY                   =   -4,
    UCC_ERR_NO_RESOURCE                 =   -5,
    UCC_ERR_NO_MESSAGE                  =   -6, /*!< General purpose return code without specific error */
    UCC_ERR_NOT_FOUND                   =   -7,
    UCC_ERR_TIMED_OUT                   =   -8,
    UCC_ERR_LAST                        = -100,
} ucc_status_t;

/**
 * @ingroup UCC_UTILS
 * @brief Routine to convert status code to string
 */

const char *ucc_status_string(ucc_status_t status);

END_C_DECLS
#endif
