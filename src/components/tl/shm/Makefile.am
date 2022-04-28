#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

bcast   =                          \
	bcast/bcast.h                  \
	bcast/bcast.c
	
reduce  =                          \
	reduce/reduce.h                \
	reduce/reduce.c

fanin   =                          \
	fanin/fanin.h                  \
	fanin/fanin.c

fanout  =                          \
	fanout/fanout.h                \
	fanout/fanout.c
	
barrier =                          \
	barrier/barrier.h              \
	barrier/barrier.c
	
allreduce =                        \
	allreduce/allreduce.h          \
	allreduce/allreduce.c

perf    =                          \
	perf/tl_shm_coll_perf_params.h \
	perf/intel/broadwell.c         \
	perf/intel/skylake.c           \
	perf/amd/rome.c

sources =                          \
	tl_shm.h                       \
	tl_shm.c                       \
	tl_shm_lib.c                   \
	tl_shm_context.c               \
	tl_shm_team.c                  \
	tl_shm_coll.h                  \
	tl_shm_coll.c                  \
	tl_shm_knomial_pattern.h       \
	tl_shm_knomial_pattern.c       \
	$(bcast)                       \
	$(reduce)                      \
	$(fanin)                       \
	$(fanout)                      \
	$(barrier)                     \
	$(allreduce)                   \
	$(perf)

module_LTLIBRARIES        = libucc_tl_shm.la
libucc_tl_shm_la_SOURCES  = $(sources)
libucc_tl_shm_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_tl_shm_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_shm_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_tl_shm_la_LIBADD   = $(SHM_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
