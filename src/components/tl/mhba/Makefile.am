#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

if TL_MHBA_ENABLED

sources =             \
	tl_mhba.h         \
	tl_mhba.c         \
	tl_mhba_lib.c     \
	tl_mhba_context.c \
	tl_mhba_team.c    \
	tl_mhba_coll.h    \
	tl_mhba_coll.c    \
	tl_mhba_ib.h      \
	tl_mhba_ib.c      \
	tl_mhba_pd.h      \
	tl_mhba_pd.c      \
	tl_mhba_mkeys.h   \
	tl_mhba_mkeys.c   \
	tl_mhba_rcache.c  \
	tl_mhba_umr.c

module_LTLIBRARIES = libucc_tl_mhba.la
libucc_tl_mhba_la_SOURCES  = $(sources)
libucc_tl_mhba_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_tl_mhba_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_mhba_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_tl_mhba_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la -libverbs -lmlx5

include $(top_srcdir)/config/module.am

endif
