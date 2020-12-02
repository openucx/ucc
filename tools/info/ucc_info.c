/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#include "ucc_info.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_datastruct.h"
#include <getopt.h>
#include <stdlib.h>

static void usage()
{
    printf("Usage: ucc_info [options]\n");
    printf("At least one of the following options has to be set:\n");
    printf("  -v Show version information\n");
    printf("  -b Show build configuration\n");
    printf("  -c Show UCC configuration\n");
    printf("  -a Show also hidden configuration\n");
    printf("  -f Display fully decorated output\n");
    printf("  -h Show this help message\n");

    printf("\n");
}
extern ucc_list_link_t ucc_config_global_list;

int main(int argc, char **argv)
{
    ucc_config_print_flags_t print_flags;
    unsigned                 print_opts;
    int                      c;
    ucc_lib_h                lib;
    ucc_lib_config_h         config;
    ucc_lib_params_t         params;
    ucc_status_t             status;
    print_flags = (ucc_config_print_flags_t)0;
    print_opts  = 0;
    while ((c = getopt(argc, argv, "vbcafh")) != -1) {
        switch (c) {
        case 'f':
            print_flags |= UCC_CONFIG_PRINT_CONFIG |
                           UCC_CONFIG_PRINT_HEADER |
                           UCC_CONFIG_PRINT_DOC;
            break;
        case 'c':
            print_flags |= UCC_CONFIG_PRINT_CONFIG;
            break;
        case 'a':
            print_flags |= UCC_CONFIG_PRINT_HIDDEN;
            break;
        case 'v':
            print_opts |= PRINT_VERSION;
            break;
        case 'b':
            print_opts |= PRINT_BUILD_CONFIG;
            break;
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return -1;
        }
    }

    if ((print_opts == 0) && (print_flags == 0)) {
        usage();
        return -2;
    }

    /* need to call ucc_init to force loading of dynamic
       ucc components */
    params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCC_THREAD_SINGLE;
    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &config)) {
        return 0;
    }

    status = ucc_init(&params, config, &lib);
    ucc_lib_config_release(config);
    if (UCC_OK != status) {
        return 0;
    }

    if (print_opts & PRINT_VERSION) {
        print_version();
    }

    if (print_opts & PRINT_BUILD_CONFIG) {
        print_build_config();
    }

    if (print_flags & UCC_CONFIG_PRINT_CONFIG) {
        ucc_config_parser_print_all_opts(stdout, "UCC_", print_flags,
                                         &ucc_config_global_list);
    }
    ucc_finalize(lib);
    return 0;
}
