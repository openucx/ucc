#
# Copyright (c) 2021 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = ucc_test_mpi

ucc_test_mpi_SOURCES = \
	test_mpi.cc        \
	main.cc            \
	buffer.cc          \
	mpi_util.cc        \
	test_case.cc       \
	test_barrier.cc    \
	test_allreduce.cc  \
	test_allgather.cc  \
	test_allgatherv.cc \
	test_bcast.cc

CXX=mpicxx
CXXFLAGS+=-I${top_builddir}/src -std=gnu++11

LDFLAGS=-L${top_builddir}/src/.libs -lucc
