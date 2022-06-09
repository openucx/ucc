#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
if TL_UCP_ENABLED

if BUILD_TL_COLL_PLUGINS
SUBDIRS = .
include makefile.coll_plugins.am
endif

barrier =                     \
	barrier/barrier.h         \
	barrier/barrier.c         \
	barrier/barrier_knomial.c

alltoall =                       \
	alltoall/alltoall.h          \
	alltoall/alltoall.c          \
	alltoall/alltoall_onesided.c \
	alltoall/alltoall_pairwise.c

alltoallv =                        \
	alltoallv/alltoallv.h          \
	alltoallv/alltoallv.c          \
	alltoallv/alltoallv_pairwise.c

bcast =                   \
	bcast/bcast.h         \
	bcast/bcast.c         \
	bcast/bcast_knomial.c \
	bcast/bcast_sag_knomial.c

allreduce =                           \
	allreduce/allreduce.h             \
	allreduce/allreduce.c             \
	allreduce/allreduce_knomial.c     \
	allreduce/allreduce_sra_knomial.c

allgather =                       \
	allgather/allgather.h         \
	allgather/allgather.c         \
	allgather/allgather_ring.c    \
	allgather/allgather_knomial.c

allgatherv =                      \
	allgatherv/allgatherv.h       \
	allgatherv/allgatherv.c       \
	allgatherv/allgatherv_ring.c

reduce =	                 \
	reduce/reduce.h          \
	reduce/reduce.c          \
	reduce/reduce_knomial.c

reduce_scatter =	                        \
	reduce_scatter/reduce_scatter.h         \
	reduce_scatter/reduce_scatter_knomial.c \
	reduce_scatter/reduce_scatter_ring.c    \
	reduce_scatter/reduce_scatter.c

reduce_scatterv =	                        \
	reduce_scatterv/reduce_scatterv.h         \
	reduce_scatterv/reduce_scatterv_ring.c    \
	reduce_scatterv/reduce_scatterv.c

gather =	                 \
	gather/gather.h          \
	gather/gather.c          \
	gather/gather_knomial.c

scatter =	                   \
	scatter/scatter.h          \
	scatter/scatter_knomial.c

fanin =	                  \
	fanin/fanin.h         \
	fanin/fanin.c

fanout =	                \
	fanout/fanout.h         \
	fanout/fanout.c

sources =                 \
	tl_ucp.h              \
	tl_ucp.c              \
	tl_ucp_lib.c          \
	tl_ucp_context.c      \
	tl_ucp_team.c         \
	tl_ucp_ep.h           \
	tl_ucp_ep.c           \
	tl_ucp_coll.c         \
	tl_ucp_service_coll.c \
	$(barrier)            \
	$(alltoall)           \
	$(alltoallv)          \
	$(allreduce)          \
	$(allgather)          \
	$(allgatherv)         \
	$(bcast)              \
	$(reduce)             \
	$(reduce_scatter)     \
	$(reduce_scatterv)    \
	$(gather)    	      \
	$(scatter)            \
	$(fanin)              \
	$(fanout)

module_LTLIBRARIES = libucc_tl_ucp.la
libucc_tl_ucp_la_SOURCES  = $(sources)
libucc_tl_ucp_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UCX_CPPFLAGS)
libucc_tl_ucp_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_ucp_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS)
libucc_tl_ucp_la_LIBADD   = $(UCX_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
