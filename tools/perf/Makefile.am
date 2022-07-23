#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = ucc_perftest

ucc_perftest_SOURCES =            \
	ucc_perftest.cc               \
	ucc_pt_config.cc              \
	ucc_pt_comm.cc                \
	ucc_pt_cuda.cc                \
	ucc_pt_rocm.cc                \
	ucc_pt_benchmark.cc           \
	ucc_pt_bootstrap_mpi.cc       \
	ucc_pt_coll.cc                \
	ucc_pt_coll_allgather.cc      \
	ucc_pt_coll_allgatherv.cc     \
	ucc_pt_coll_allreduce.cc      \
	ucc_pt_coll_alltoall.cc       \
	ucc_pt_coll_alltoallv.cc      \
	ucc_pt_coll_barrier.cc        \
	ucc_pt_coll_bcast.cc          \
	ucc_pt_coll_gather.cc         \
	ucc_pt_coll_gatherv.cc        \
	ucc_pt_coll_reduce.cc         \
	ucc_pt_coll_reduce_scatter.cc \
	ucc_pt_coll_scatter.cc        \
	ucc_pt_coll_scatterv.cc

CXX=$(MPICXX)
LD=$(MPICXX)
ucc_perftest_CPPFLAGS = $(BASE_CPPFLAGS)
ucc_perftest_CXXFLAGS = -std=gnu++11 $(BASE_CXXFLAGS)
ucc_perftest_LDFLAGS = -Wl,--rpath-link=${UCS_LIBDIR}
ucc_perftest_LDADD = $(UCC_TOP_BUILDDIR)/src/libucc.la -ldl
