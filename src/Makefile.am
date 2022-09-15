#
# Copyright (C) Mellanox Technologies Ltd. 2020-2022.  ALL RIGHTS RESERVED.
#

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
include utils/Makefile.am

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
	$(utils_noinst_headers)           \
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
	components/topo/ucc_topo.h        \
	components/topo/ucc_sbgp.h

libucc_la_SOURCES =                   \
	$(utils_libucc_la_sources)        \
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
