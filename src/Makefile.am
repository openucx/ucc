#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#
SUBDIRS = cl/basic

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

noinst_HEADERS =             \
	core/ucc_global_opts.h   \
	core/ucc_lib.h           \
	core/ucc_context.h       \
	utils/ucc_compiler_def.h \
	utils/ucc_log.h          \
	utils/ucc_parser.h       \
	utils/ucc_component.h    \
	utils/ucc_datastruct.h   \
	utils/ucc_math.h         \
	cl/ucc_cl.h              \
	cl/ucc_cl_log.h          \
	cl/ucc_base_iface.h

libucc_la_SOURCES =        \
	core/ucc_lib.c         \
	core/ucc_constructor.c \
	core/ucc_global_opts.c \
	core/ucc_version.c     \
	core/ucc_context.c     \
	utils/ucc_component.c  \
	cl/ucc_cl.c            \
	cl/ucc_base_iface.c

libucc_ladir = $(includedir)
