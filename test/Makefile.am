#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = test_lib_init test_context_create test_team_create

test_lib_init_SOURCES = test_lib_init.c

test_context_create_SOURCES = test_context_create.c

test_team_create_SOURCES = test_team_create.c
test_team_create_LDFLAGS = -lmpi

CFLAGS+=-I${includedir}
LDFLAGS=-L$(libdir) -lucc
