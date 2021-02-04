#
# Copyright (c) 2021 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = test_mpi_barrier

test_mpi_barrier_SOURCES=test_mpi_barrier.c test_mpi.c

CC=mpicc
CFLAGS+=-I${top_builddir}/src -std=c11

LDFLAGS=-L${top_builddir}/src/.libs -lucc
