#
# Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
#

cl_dirs = components/cl/basic
tl_dirs =
mc_dirs = components/mc/cpu

if HAVE_UCX
tl_dirs += components/tl/ucp
endif

if HAVE_CUDA
mc_dirs += components/mc/cuda
endif

if HAVE_NCCL
tl_dirs += components/tl/nccl
endif

if HAVE_DPU
tl_dirs += components/tl/dpu
endif

SUBDIRS = . $(cl_dirs) $(tl_dirs) $(mc_dirs)
lib_LTLIBRARIES  = libucc.la
noinst_LIBRARIES =

libucc_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_la_CFLAGS   = -c $(BASE_CFLAGS)
libucc_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed -pthread

nobase_dist_libucc_la_HEADERS =	\
	ucc/api/ucc.h                   \
	ucc/api/ucc_def.h               \
	ucc/api/ucc_version.h           \
	ucc/api/ucc_status.h

noinst_HEADERS =                      \
	core/ucc_global_opts.h            \
	core/ucc_lib.h                    \
	core/ucc_context.h                \
	core/ucc_mc.h                     \
	core/ucc_team.h                   \
	core/ucc_ee.h                     \
	core/ucc_progress_queue.h         \
	schedule/ucc_schedule.h           \
	coll_score/ucc_coll_score.h       \
	utils/ucc_compiler_def.h          \
	utils/ucc_log.h                   \
	utils/ucc_parser.h                \
	utils/ucc_component.h             \
	utils/ucc_datastruct.h            \
	utils/ucc_math.h                  \
	utils/ucc_coll_utils.h            \
	utils/ucc_list.h                  \
	utils/ucc_string.h                \
	utils/ucc_queue.h                 \
	utils/ucc_proc_info.h             \
	utils/khash.h                     \
	utils/ucc_spinlock.h              \
	utils/ucc_mpool.h                 \
	utils/profile/ucc_profile.h       \
	utils/profile/ucc_profile_on.h    \
	utils/profile/ucc_profile_off.h   \
	components/base/ucc_base_iface.h  \
	components/cl/ucc_cl.h            \
	components/cl/ucc_cl_log.h        \
	components/cl/ucc_cl_type.h       \
	components/tl/ucc_tl.h            \
	components/tl/ucc_tl_log.h        \
	components/mc/base/ucc_mc_base.h  \
	components/mc/ucc_mc_log.h        \
	coll_patterns/recursive_knomial.h \
	coll_patterns/sra_knomial.h

libucc_la_SOURCES =                  \
	core/ucc_lib.c                   \
	core/ucc_constructor.c           \
	core/ucc_global_opts.c           \
	core/ucc_version.c               \
	core/ucc_context.c               \
	core/ucc_mc.c                    \
	core/ucc_team.c                  \
	core/ucc_ee.c                    \
	core/ucc_coll.c                  \
	core/ucc_progress_queue.c        \
	core/ucc_progress_queue_st.c     \
	core/ucc_progress_queue_mt.c     \
	schedule/ucc_schedule.c          \
	coll_score/ucc_coll_score.c      \
	coll_score/ucc_coll_score_map.c  \
	utils/ucc_component.c            \
	utils/ucc_status.c               \
	utils/ucc_mpool.c                \
	utils/ucc_math.c                 \
	utils/ucc_proc_info.c            \
	utils/ucc_string.c               \
	utils/ucc_coll_utils.c           \
	utils/ucc_parser.c               \
	utils/profile/ucc_profile.c      \
	components/base/ucc_base_iface.c \
	components/cl/ucc_cl.c           \
	components/tl/ucc_tl.c           \
	components/mc/base/ucc_mc_base.c

libucc_ladir = $(includedir)
