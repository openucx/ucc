#
# Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
#

cl_dirs = components/cl/basic
tl_dirs =
mc_dirs = components/mc/cpu

if HAVE_UCX
tl_dirs += components/tl/ucp
endif

SUBDIRS = $(cl_dirs) $(tl_dirs) $(mc_dirs)
lib_LTLIBRARIES  = libucc.la
noinst_LIBRARIES =

libucc_la_CPPFLAGS = $(AM_CPPFLAGS)
libucc_la_CFLAGS   = -c
libucc_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed

nobase_dist_libucc_la_HEADERS = 	\
	ucc/api/ucc.h                   \
	ucc/api/ucc_def.h               \
	ucc/api/ucc_version.h           \
	ucc/api/ucc_status.h

noinst_HEADERS =                     \
	core/ucc_global_opts.h           \
	core/ucc_lib.h                   \
	core/ucc_context.h               \
	core/ucc_mc.h                    \
	utils/ucc_compiler_def.h         \
	utils/ucc_log.h                  \
	utils/ucc_parser.h               \
	utils/ucc_component.h            \
	utils/ucc_datastruct.h           \
	utils/ucc_math.h                 \
	components/base/ucc_base_iface.h \
	components/cl/ucc_cl.h           \
	components/cl/ucc_cl_log.h       \
	components/cl/ucc_cl_type.h      \
	components/tl/ucc_tl.h           \
	components/tl/ucc_tl_log.h       \
	components/mc/base/ucc_mc_base.h \
	components/mc/ucc_mc_log.h

libucc_la_SOURCES =                  \
	core/ucc_lib.c                   \
	core/ucc_constructor.c           \
	core/ucc_global_opts.c           \
	core/ucc_version.c               \
	core/ucc_context.c               \
	core/ucc_mc.c                    \
	utils/ucc_component.c            \
	components/base/ucc_base_iface.c \
	components/cl/ucc_cl.c           \
	components/tl/ucc_tl.c           \
	components/mc/base/ucc_mc_base.c

libucc_ladir = $(includedir)
