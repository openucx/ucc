#
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_DOCA_UROM

sources =                \
	common/urom_ucc.h    \
	dpu/worker_ucc_p2p.c \
	dpu/worker_ucc.h     \
	dpu/worker_ucc.c

plugindir = $(moduledir)/doca_plugins

plugin_LTLIBRARIES         = libucc_doca_urom_plugin.la
libucc_doca_urom_plugin_la_SOURCES  = $(sources)
libucc_doca_urom_plugin_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UCX_CPPFLAGS) $(DOCA_UROM_CPPFLAGS)
libucc_doca_urom_plugin_la_CFLAGS   = $(BASE_CFLAGS)
libucc_doca_urom_plugin_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS) $(DOCA_UROM_LDFLAGS)
libucc_doca_urom_plugin_la_LIBADD   = $(UCX_LIBADD) $(DOCA_UROM_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

endif
