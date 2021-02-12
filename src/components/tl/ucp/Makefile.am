#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#

barrier =                     \
	barrier/barrier.h         \
	barrier/barrier.c         \
	barrier/barrier_knomial.c

alltoall =                       \
	alltoall/alltoall.h          \
	alltoall/alltoall.c          \
	alltoall/alltoall_pairwise.c

sources =            \
	tl_ucp.h         \
	tl_ucp.c         \
	tl_ucp_lib.c     \
	tl_ucp_context.c \
	tl_ucp_team.c    \
	tl_ucp_ep.h      \
	tl_ucp_ep.c      \
	tl_ucp_addr.h    \
	tl_ucp_addr.c    \
	tl_ucp_coll.c    \
	$(barrier)       \
    $(alltoall)

module_LTLIBRARIES = libucc_tl_ucp.la
libucc_tl_ucp_la_SOURCES  = $(sources)
libucc_tl_ucp_la_CPPFLAGS = $(AM_CPPFLAGS) $(UCX_CPPFLAGS)
libucc_tl_ucp_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS)
libucc_tl_ucp_la_LIBADD   = $(UCX_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
