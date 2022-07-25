/**
 * Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_info.h"
#include "utils/ucc_compiler_def.h"

void print_version()
{
    printf("# UCC version=%s revision %s\n", UCC_VERSION_STRING,
           UCC_GIT_REVISION);
}

void print_build_config()
{
    typedef struct {
        const char *name;
        const char *value;
    } config_var_t;
    static config_var_t config_vars[] = {
        #include <build_config.h>
        {NULL, NULL}
    };
    config_var_t *var;

    for (var = config_vars; var->name != NULL; ++var) {
        printf("#define %-25s %s\n", var->name, var->value);
    }
}
