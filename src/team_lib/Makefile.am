#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

SUBDIRS = basic

noinst_LTLIBRARIES  = libucc_tl.la

libucc_tl_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_tl_la_LDFLAGS  = -ldl -version-info $(SOVERSION) --no-undefined

libucc_tl_la_SOURCES = \
	ucc_tl.h \
	ucc_tl.c

