#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_IBVERBS

sources =        \
	ucc_sysinfo_ib.h \
	ucc_sysinfo_ib.c

module_LTLIBRARIES            = libucc_sysinfo_ib.la
libucc_sysinfo_ib_la_SOURCES   = $(sources)
libucc_sysinfo_ib_la_CPPFLAGS  = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) \
                                 $(IBVERBS_CPPFLAGS)
libucc_sysinfo_ib_la_CFLAGS    = $(BASE_CFLAGS)
libucc_sysinfo_ib_la_LDFLAGS   = -version-info $(SOVERSION) --as-needed \
                                 $(IBVERBS_LDFLAGS)
libucc_sysinfo_ib_la_LIBADD    = $(IBVERBS_LIBADD)               \
                                 $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
endif
