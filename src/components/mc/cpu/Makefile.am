#
# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

sources =    \
	mc_cpu.h \
	mc_cpu.c

module_LTLIBRARIES        = libucc_mc_cpu.la
libucc_mc_cpu_la_SOURCES  = $(sources)
libucc_mc_cpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_mc_cpu_la_CFLAGS   = $(BASE_CFLAGS)
libucc_mc_cpu_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_mc_cpu_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
