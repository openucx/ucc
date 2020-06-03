/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <api/ucc.h>
#include <stdio.h>

int main(int argc, char **argv) {
    ucc_lib_config_t *config;
    ucc_lib_h lib;
    ucc_status_t status;
    ucc_lib_params_t params = {
        .field_mask = 0,
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

    ucc_lib_cleanup(lib);
    return 0;
error:
    return -1;
}
