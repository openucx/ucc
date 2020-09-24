/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <api/ucc.h>
#include <stdio.h>

int main(int argc, char **argv) {
    ucc_lib_config_h config;
    ucc_context_config_h ctx_config;
    ucc_lib_h lib;
    ucc_context_h ucc_ctx;
    ucc_status_t status;
    ucc_context_params_t ctx_params = {
        .mask = 0,
    };

    ucc_lib_params_t params = {
        .mask = 0,
    };

    status = ucc_lib_config_read("TEST", NULL, &config);
    if (UCC_OK != status) {
        goto error;
    }

    status = ucc_lib_init(&params, config, &lib);
    if (UCC_OK != status) {
        goto error;
    }

    ucc_lib_config_release(config);

    status = ucc_context_config_read(lib, NULL, &ctx_config);
    if (UCC_OK != status) {
        goto error;
    }

    status = ucc_context_create(lib, &ctx_params, ctx_config, &ucc_ctx);
    if (UCC_OK != status) {
        goto error;
    }
    ucc_context_config_release(ctx_config);

    ucc_context_destroy(ucc_ctx);
    ucc_lib_cleanup(lib);
    return 0;
error:
    return -1;
}
