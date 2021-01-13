#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#

sources =              \
	cl_basic.h         \
	cl_basic.c         \
	cl_basic_lib.c     \
	cl_basic_context.c \
	cl_basic_team.c

module_LTLIBRARIES          = libucc_cl_basic.la
libucc_cl_basic_la_SOURCES  = $(sources)
libucc_cl_basic_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_cl_basic_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed

include $(top_srcdir)/config/module.am
