/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#include "ucc_info.h"
#include "utils/ucc_parser.h"
#include <getopt.h>
#include <stdlib.h>

static void usage()
{
    printf("Usage: ucc_info [options]\n");
    printf("At least one of the following options has to be set:\n");
    printf("  -v Show version information\n");
    printf("  -h Show this help message\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    unsigned print_opts;
    int      c;

    print_opts = 0;
    while ((c = getopt(argc, argv, "vh")) != -1) {
        switch (c) {
        case 'v':
            print_opts |= PRINT_VERSION;
            break;
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return -1;
        }
    }

    if (print_opts == 0) {
        usage();
        return -2;
    }

    if (print_opts & PRINT_VERSION) {
        print_version();
    }

    return 0;
}
