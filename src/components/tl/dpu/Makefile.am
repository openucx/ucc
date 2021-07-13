#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

sources =				\
	tl_dpu.c			\
	tl_dpu.h			\
	tl_dpu_lib.c		\
	tl_dpu_context.c	\
	tl_dpu_team.c		\
	tl_dpu_coll.c		\
	tl_dpu_coll.h

module_LTLIBRARIES = libucc_tl_dpu.la
libucc_tl_dpu_la_SOURCES	= $(sources)
libucc_tl_dpu_la_CPPFLAGS	= $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UCX_CPPFLAGS) $(DPU_CPPFLAGS) 
libucc_tl_dpu_la_CFLAGS		= $(BASE_CFLAGS)
libucc_tl_dpu_la_LDFLAGS	= -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS) $(DPU_LDFLAGS)
libucc_tl_dpu_la_LIBADD		= $(UCX_LIBADD) $(DPU_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am