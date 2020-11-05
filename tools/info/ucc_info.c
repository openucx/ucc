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
    printf("  -h Show this help message\n");
    printf("\n");
}
extern ucc_list_link_t ucc_config_global_list;

int main(int argc, char **argv)
{
    ucc_config_print_flags_t print_flags;
    unsigned print_opts;
    int      c;

    print_flags = (ucc_config_print_flags_t)0;
    print_opts  = 0;
    while ((c = getopt(argc, argv, "vbch")) != -1) {
        switch (c) {
        case 'c':
            print_flags |= UCC_CONFIG_PRINT_CONFIG;
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

    return 0;
}
