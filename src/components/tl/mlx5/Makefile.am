#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

if TL_MLX5_ENABLED

sources =             \
	tl_mlx5.h         \
	tl_mlx5.c         \
	tl_mlx5_lib.c     \
	tl_mlx5_context.c \
	tl_mlx5_team.c    \
	tl_mlx5_coll.h    \
	tl_mlx5_coll.c    \
	tl_mlx5_ib.h      \
	tl_mlx5_ib.c      \
	tl_mlx5_pd.h      \
	tl_mlx5_pd.c      \
	tl_mlx5_mkeys.h   \
	tl_mlx5_mkeys.c   \
	tl_mlx5_rcache.c  \
	tl_mlx5_umr.c     \
	tl_mlx5_inline.h

module_LTLIBRARIES = libucc_tl_mlx5.la
libucc_tl_mlx5_la_SOURCES  = $(sources)
libucc_tl_mlx5_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_tl_mlx5_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_mlx5_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_tl_mlx5_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la -libverbs -lmlx5

include $(top_srcdir)/config/module.am

endif
