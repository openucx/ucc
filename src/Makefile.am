#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.

cl_dirs = components/cl/basic \
		  components/cl/hier

mc_dirs = components/mc/cpu
ec_dirs = components/ec/cpu

if HAVE_CUDA
mc_dirs += components/mc/cuda
ec_dirs += components/ec/cuda
endif

if HAVE_ROCM
mc_dirs += components/mc/rocm
ec_dirs += components/ec/rocm
endif

SUBDIRS = . $(cl_dirs) $(mc_dirs) $(ec_dirs)

include components/tl/makefile.am

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
	core/ucc_team.h                   \
	core/ucc_ee.h                     \
	core/ucc_progress_queue.h         \
	core/ucc_service_coll.h           \
	core/ucc_dt.h	                  \
	schedule/ucc_schedule.h           \
	schedule/ucc_schedule_pipelined.h \
	coll_score/ucc_coll_score.h       \
	utils/arch/aarch64/cpu.h          \
	utils/arch/ppc64/cpu.h            \
	utils/arch/riscv64/cpu.h          \
	utils/arch/x86_64/cpu.h           \
	utils/arch/cpu.h                  \
	utils/arch/cuda_def.h             \
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
	utils/ini.h                       \
	utils/ucc_spinlock.h              \
	utils/ucc_mpool.h                 \
	utils/ucc_rcache.h                \
	utils/profile/ucc_profile.h       \
	utils/profile/ucc_profile_on.h    \
	utils/profile/ucc_profile_off.h   \
	utils/ucc_time.h                  \
	utils/ucc_sys.h                   \
	utils/ucc_assert.h                \
	components/base/ucc_base_iface.h  \
	components/cl/ucc_cl.h            \
	components/cl/ucc_cl_log.h        \
	components/cl/ucc_cl_type.h       \
	components/tl/ucc_tl.h            \
	components/tl/ucc_tl_log.h        \
	components/mc/ucc_mc.h            \
	components/mc/base/ucc_mc_base.h  \
	components/mc/ucc_mc_log.h        \
	components/ec/ucc_ec.h            \
	components/ec/base/ucc_ec_base.h  \
	components/ec/ucc_ec_log.h        \
	coll_patterns/recursive_knomial.h \
	coll_patterns/sra_knomial.h       \
	coll_patterns/bruck_alltoall.h    \
	coll_patterns/two_tree.h          \
	components/topo/ucc_topo.h        \
	components/topo/ucc_sbgp.h

libucc_la_SOURCES =                   \
	core/ucc_lib.c                    \
	core/ucc_constructor.c            \
	core/ucc_global_opts.c            \
	core/ucc_version.c                \
	core/ucc_context.c                \
	core/ucc_team.c                   \
	core/ucc_ee.c                     \
	core/ucc_coll.c                   \
	core/ucc_progress_queue.c         \
	core/ucc_progress_queue_st.c      \
	core/ucc_progress_queue_mt.c      \
	core/ucc_service_coll.c           \
	core/ucc_dt.c                     \
	schedule/ucc_schedule.c           \
	schedule/ucc_schedule_pipelined.c \
	coll_score/ucc_coll_score.c       \
	coll_score/ucc_coll_score_map.c   \
	utils/ini.c                       \
	utils/ucc_component.c             \
	utils/ucc_datastruct.c            \
	utils/ucc_status.c                \
	utils/ucc_mpool.c                 \
	utils/ucc_math.c                  \
	utils/ucc_proc_info.c             \
	utils/ucc_string.c                \
	utils/ucc_coll_utils.c            \
	utils/ucc_parser.c                \
	utils/profile/ucc_profile.c       \
	utils/ucc_sys.c                   \
	utils/arch/x86_64/cpu.c           \
	utils/arch/aarch64/cpu.c          \
	utils/ucc_assert.c                \
	components/base/ucc_base_iface.c  \
	components/cl/ucc_cl.c            \
	components/tl/ucc_tl.c            \
	components/mc/ucc_mc.c            \
	components/mc/base/ucc_mc_base.c  \
	components/ec/ucc_ec.c            \
	components/ec/base/ucc_ec_base.c  \
	components/topo/ucc_topo.c        \
	components/topo/ucc_sbgp.c

libucc_ladir = $(includedir)
