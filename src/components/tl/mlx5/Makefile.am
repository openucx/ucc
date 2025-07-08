#
# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if TL_MLX5_ENABLED

alltoall =                      \
	alltoall/alltoall.h         \
	alltoall/alltoall.c         \
	alltoall/alltoall_mkeys.h   \
	alltoall/alltoall_mkeys.c   \
	alltoall/alltoall_inline.h  \
	alltoall/alltoall_coll.c

mcast =                                         \
	mcast/tl_mlx5_mcast_context.c               \
	mcast/tl_mlx5_mcast.h                       \
	mcast/tl_mlx5_mcast_coll.c                  \
	mcast/tl_mlx5_mcast_coll.h                  \
	mcast/tl_mlx5_mcast_rcache.h                \
	mcast/tl_mlx5_mcast_rcache.c                \
	mcast/p2p/ucc_tl_mlx5_mcast_p2p.h           \
	mcast/p2p/ucc_tl_mlx5_mcast_p2p.c           \
	mcast/tl_mlx5_mcast_progress.h              \
	mcast/tl_mlx5_mcast_progress.c              \
	mcast/tl_mlx5_mcast_helper.h                \
	mcast/tl_mlx5_mcast_helper.c                \
	mcast/tl_mlx5_mcast_service_coll.h          \
	mcast/tl_mlx5_mcast_service_coll.c          \
	mcast/tl_mlx5_mcast_one_sided_reliability.h \
	mcast/tl_mlx5_mcast_one_sided_reliability.c \
	mcast/tl_mlx5_mcast_one_sided_progress.h    \
	mcast/tl_mlx5_mcast_one_sided_progress.c    \
	mcast/tl_mlx5_mcast_allgather.h             \
	mcast/tl_mlx5_mcast_allgather.c             \
	mcast/tl_mlx5_mcast_hca_copy.h              \
	mcast/tl_mlx5_mcast_hca_copy.c              \
	mcast/tl_mlx5_mcast_team.c

sources =             \
	tl_mlx5.h         \
	tl_mlx5.c         \
	tl_mlx5_lib.c     \
	tl_mlx5_context.c \
	tl_mlx5_team.c    \
	tl_mlx5_coll.h    \
	tl_mlx5_coll.c    \
	tl_mlx5_ib.h	  \
	tl_mlx5_ib.c      \
	tl_mlx5_wqe.h	  \
	tl_mlx5_wqe.c	  \
	tl_mlx5_pd.h      \
	tl_mlx5_pd.c      \
	tl_mlx5_rcache.c  \
	tl_mlx5_dm.c      \
	tl_mlx5_dm.h      \
	$(mcast)          \
	$(alltoall)

module_LTLIBRARIES = libucc_tl_mlx5.la
libucc_tl_mlx5_la_SOURCES  = $(sources)
libucc_tl_mlx5_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_tl_mlx5_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_mlx5_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_tl_mlx5_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la $(IBVERBS_LIBADD) $(MLX5DV_LIBADD) $(RDMACM_LIBADD)

include $(top_srcdir)/config/module.am

endif
