/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_DEF_H_
#define UCC_DEF_H_

#include <stddef.h>
#include <stdint.h>

typedef struct ucc_lib* ucc_lib_h;

/**
 * @ingroup UCC_LIB_CONFIG
 * @brief UCC configuration descriptor
 *
 * This descriptor defines the configuration for @ref ucc_lib_h
 * "UCC team library". The configuration is loaded from the run-time
 * environment (using configuration files of environment variables)
 * using @ref ucc_lib_config_read "ucc_lib_config_read" routine and can be printed
 * using @ref ucc_lib_config_print "ucc_lib_config_print" routine. In addition,
 * application is responsible to release the descriptor using
 * @ref ucc_lib_config_release "ucc_lib_config_release" routine.
 */
typedef struct ucc_lib_config*     ucc_lib_config_h;
typedef struct ucc_context_config* ucc_context_config_h;
typedef struct ucc_context*        ucc_context_h;
typedef struct ucc_team*           ucc_team_h;
#endif
