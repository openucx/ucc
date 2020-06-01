#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

SUBDIRS = core


lib_LTLIBRARIES  = libucc.la
noinst_LIBRARIES =

libucc_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_la_CFLAGS   =
libucc_la_LDFLAGS  = -ldl -version-info $(SOVERSION) --no-undefined

nobase_dist_libucc_la_HEADERS = \
	api/ucc.h                   \
	api/ucc_def.h               \
	api/ucc_version.h           \
	api/ucc_status.h

noinst_HEADERS =      \
	team_lib/ucc_tl.h \
	utils/ucc_log.h

libucc_la_SOURCES =

libucc_la_LIBADD = \
	core/libucc_core.la

libucc_ladir = $(includedir)

install-exec-hook:
	cp -f $(UCC_TOP_BUILDDIR)/src/libucc.la $(libdir)/
	perl -pi -e "s/installed=no/installed=yes/" $(libdir)/libucc.la
