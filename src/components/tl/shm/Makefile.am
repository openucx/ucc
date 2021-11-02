#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

bcast =              \
	bcast/bcast.h    \
	bcast/bcast.c
	
reduce =             \
	reduce/reduce.h  \
	reduce/reduce.c

sources =                    \
	tl_shm.h                 \
	tl_shm.c                 \
	tl_shm_lib.c             \
	tl_shm_context.c         \
	tl_shm_team.c            \
	tl_shm_coll.h            \
	tl_shm_coll.c            \
	tl_shm_knomial_pattern.h \
	$(bcast)                 \
	$(reduce)

module_LTLIBRARIES = libucc_tl_shm.la
libucc_tl_shm_la_SOURCES  = $(sources)
libucc_tl_shm_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(SHM_CPPFLAGS)
libucc_tl_shm_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_shm_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(SHM_LDFLAGS)
libucc_tl_shm_la_LIBADD   = $(SHM_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
