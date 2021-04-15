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
	test_bcast.cc      \
	test_alltoall.cc   \
	test_alltoallv.cc

CXX=mpicxx
ucc_test_mpi_CXXFLAGS=-I${top_builddir}/src -std=gnu++11

ucc_test_mpi_LDFLAGS=-L${top_builddir}/src/.libs -lucc

if HAVE_CUDA
ucc_test_mpi_CXXFLAGS+=$(CUDA_CPPFLAGS) -DUCC_TEST_WITH_CUDA
ucc_test_mpi_LDFLAGS+=$(CUDA_LDFLAGS) $(CUDA_LIBS)
endif
