#
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_UCP_MEMH_PACK

sources =                           \
	cl_doca_urom.h                  \
	cl_doca_urom.c                  \
	cl_doca_urom_lib.c              \
	cl_doca_urom_context.c          \
	cl_doca_urom_team.c             \
	cl_doca_urom_common.c           \
	cl_doca_urom_common.h           \
	cl_doca_urom_worker_ucc.c       \
	cl_doca_urom_worker_ucc.h       \
	cl_doca_urom_coll.c

module_LTLIBRARIES         = libucc_cl_doca_urom.la
libucc_cl_doca_urom_la_SOURCES  = $(sources)
libucc_cl_doca_urom_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(DOCA_UROM_CPPFLAGS) -I$(top_srcdir)/contrib/doca_urom_ucc_plugin/common
libucc_cl_doca_urom_la_CFLAGS   = $(BASE_CFLAGS)
libucc_cl_doca_urom_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(DOCA_UROM_LDFLAGS)
libucc_cl_doca_urom_la_LIBADD   = $(DOCA_UROM_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif