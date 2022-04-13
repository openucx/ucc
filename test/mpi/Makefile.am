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
	test_mpi.cc             \
	main.cc                 \
	buffer.cc               \
	mpi_util.cc             \
	test_case.cc            \
	test_barrier.cc         \
	test_allreduce.cc       \
	test_allgather.cc       \
	test_allgatherv.cc      \
	test_bcast.cc           \
	test_alltoall.cc        \
	test_alltoallv.cc       \
	test_reduce_scatter.cc  \
	test_reduce_scatterv.cc \
	test_reduce.cc          \
	test_gather.cc

CXX=$(MPICXX)
LD=$(MPICXX)
ucc_test_mpi_CPPFLAGS=$(BASE_CPPFLAGS)
ucc_test_mpi_CXXFLAGS=$(BASE_CXXFLAGS) -std=gnu++11
ucc_test_mpi_LDADD=$(UCC_TOP_BUILDDIR)/src/libucc.la

if HAVE_CUDA
ucc_test_mpi_CPPFLAGS+=$(CUDA_CPPFLAGS)
ucc_test_mpi_LDFLAGS=$(CUDA_LDFLAGS)
ucc_test_mpi_LDADD+=$(CUDA_LIBS)
endif
