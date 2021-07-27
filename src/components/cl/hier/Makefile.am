#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#

sources =              \
	cl_hier.h         \
	cl_hier.c         \
	cl_hier_lib.c     \
	cl_hier_context.c \
	cl_hier_team.c    \
	cl_hier_coll.c

module_LTLIBRARIES          = libucc_cl_hier.la
libucc_cl_hier_la_SOURCES  = $(sources)
libucc_cl_hier_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_cl_hier_la_CFLAGS   = $(BASE_CFLAGS)
libucc_cl_hier_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_cl_hier_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
