#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

allgatherv =                         \
	allgatherv/unpack.h              \
	allgatherv/unpack.c              \
	allgatherv/allgatherv.h          \
	allgatherv/allgatherv.c

allreduce =                          \
	allreduce/allreduce.h            \
	allreduce/allreduce.c            \
	allreduce/allreduce_rab.c        \
	allreduce/allreduce_split_rail.c

alltoallv =                          \
	alltoallv/alltoallv.h            \
	alltoallv/alltoallv.c

alltoall =                           \
	alltoall/alltoall.h              \
	alltoall/alltoall.c

barrier =                            \
	barrier/barrier.h                \
	barrier/barrier.c

bcast =                              \
	bcast/bcast.h                    \
	bcast/bcast.c                    \
	bcast/bcast_2step.c

reduce =                             \
	reduce/reduce.h                  \
	reduce/reduce.c                  \
	reduce/reduce_2step.c

sources =                            \
	cl_hier.h                        \
	cl_hier.c                        \
	cl_hier_lib.c                    \
	cl_hier_context.c                \
	cl_hier_team.c                   \
	cl_hier_coll.c                   \
	cl_hier_coll.h                   \
	$(allgatherv)                    \
	$(allreduce)                     \
	$(alltoallv)                     \
	$(alltoall)                      \
	$(barrier)                       \
	$(bcast)                         \
	$(reduce)

module_LTLIBRARIES         = libucc_cl_hier.la
libucc_cl_hier_la_SOURCES  = $(sources)
libucc_cl_hier_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_cl_hier_la_CFLAGS   = $(BASE_CFLAGS)
libucc_cl_hier_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_cl_hier_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
