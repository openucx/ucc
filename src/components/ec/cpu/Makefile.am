#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

sources =    \
	ec_cpu.h \
	ec_cpu.c

module_LTLIBRARIES        = libucc_ec_cpu.la
libucc_ec_cpu_la_SOURCES  = $(sources)
libucc_ec_cpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_ec_cpu_la_CFLAGS   = $(BASE_CFLAGS)
libucc_ec_cpu_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_ec_cpu_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
