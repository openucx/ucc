#
# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
#

if TL_CUDA_ENABLED

SUBDIRS = kernels

allgather =                      \
	allgather/allgather.h        \
	allgather/allgather.c        \
	allgather/allgather_ring.c   \
	allgather/allgather_linear.c

allgatherv =                       \
	allgatherv/allgatherv.h        \
	allgatherv/allgatherv.c        \
	allgatherv/allgatherv_ring.c   \
	allgatherv/allgatherv_linear.c

alltoall =                 \
	alltoall/alltoall.h    \
	alltoall/alltoall.c    \
	alltoall/alltoall_ce.c

alltoallv =                  \
	alltoallv/alltoallv.h    \
	alltoallv/alltoallv.c    \
	alltoallv/alltoallv_ce.c

bcast =                  \
	bcast/bcast.h        \
	bcast/bcast.c        \
	bcast/bcast_linear.c

reduce_scatter =                           \
	reduce_scatter/reduce_scatter.h        \
	reduce_scatter/reduce_scatter.c        \
	reduce_scatter/reduce_scatter_ring.c   \
	reduce_scatter/reduce_scatter_linear.c

reduce_scatterv =                            \
	reduce_scatterv/reduce_scatterv.h        \
	reduce_scatterv/reduce_scatterv.c        \
	reduce_scatterv/reduce_scatterv_ring.c   \
	reduce_scatterv/reduce_scatterv_linear.c

allreduce =                    \
	allreduce/allreduce.h      \
	allreduce/allreduce.c

if TL_CUDA_NVLS_ENABLED
reduce_scatter += reduce_scatter/reduce_scatter_nvls.c
reduce_scatterv += reduce_scatterv/reduce_scatterv_nvls.c
allreduce += allreduce/allreduce_nvls.c
endif

sources =               \
	tl_cuda.h           \
	tl_cuda.c           \
	tl_cuda_lib.c       \
	tl_cuda_context.c   \
	tl_cuda_team.c      \
	tl_cuda_coll.c      \
	tl_cuda_cache.c     \
	tl_cuda_topo.c      \
	tl_cuda_team_topo.c \
	$(allgather)        \
	$(allgatherv)       \
	$(alltoall)         \
	$(alltoallv)        \
	$(bcast)            \
	$(allreduce)        \
	$(reduce_scatter)   \
	$(reduce_scatterv)

if TL_CUDA_NVLS_ENABLED
sources += tl_cuda_nvls.h tl_cuda_nvls.c
endif

module_LTLIBRARIES = libucc_tl_cuda.la
libucc_tl_cuda_la_SOURCES  = $(sources)
libucc_tl_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_tl_cuda_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_tl_cuda_la_LIBADD   = $(CUDA_LIBS) $(NVML_LIBS) $(UCC_TOP_BUILDDIR)/src/libucc.la kernels/libucc_tl_cuda_kernels.la

include $(top_srcdir)/config/module.am

endif
