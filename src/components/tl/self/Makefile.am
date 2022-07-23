#
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
#

if TL_SELF_ENABLED
sources =               	\
	tl_self.h       	\
	tl_self.c       	\
	tl_self_coll.c       	\
	tl_self_context.c 	\
	tl_self_lib.c       	\
	tl_self_team.c 


module_LTLIBRARIES = libucc_tl_self.la
libucc_tl_self_la_SOURCES  = $(sources)
libucc_tl_self_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_tl_self_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_self_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_tl_self_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
