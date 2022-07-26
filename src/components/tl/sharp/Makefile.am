#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if TL_SHARP_ENABLED

sources =             \
        tl_sharp.h         \
        tl_sharp.c         \
        tl_sharp_lib.c     \
        tl_sharp_context.c \
        tl_sharp_team.c    \
        tl_sharp_coll.h    \
        tl_sharp_coll.c

module_LTLIBRARIES = libucc_tl_sharp.la
libucc_tl_sharp_la_SOURCES  = $(sources)
libucc_tl_sharp_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(SHARP_CPPFLAGS)
libucc_tl_sharp_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_sharp_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(SHARP_LDFLAGS)
libucc_tl_sharp_la_LIBADD   = $(SHARP_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
