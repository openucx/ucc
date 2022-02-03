#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

if TL_CUDA_ENABLED
alltoall =                       \
	alltoall/alltoall.h          \
	alltoall/alltoall.c          \
	alltoall/alltoall_ce.c

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
	$(alltoall)

module_LTLIBRARIES = libucc_tl_cuda.la
libucc_tl_cuda_la_SOURCES  = $(sources)
libucc_tl_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_tl_cuda_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_tl_cuda_la_LIBADD   = $(CUDA_LIBS) $(NVML_LIBS) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
