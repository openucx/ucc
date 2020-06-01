#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

noinst_LTLIBRARIES  = libucc_core.la

libucc_core_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_core_la_LDFLAGS  = -ldl -version-info $(SOVERSION) --no-undefined

libucc_core_la_SOURCES = \
	ucc_constructor.c \
	ucc_lib.c         \
	ucc_lib.h         \
	ucc_global_opts.c \
	ucc_global_opts.h
