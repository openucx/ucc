#
# Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
#

sources =    \
	mc_cpu.h \
	mc_cpu_reduce.h \
	mc_cpu.c

module_LTLIBRARIES        = libucc_mc_cpu.la
libucc_mc_cpu_la_SOURCES  = $(sources)
libucc_mc_cpu_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_mc_cpu_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed

include $(top_srcdir)/config/module.am
