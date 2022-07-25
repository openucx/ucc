#
# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

sources =                                              \
	mc_cpu.h                                           \
	mc_cpu.c                                           \
	reduce/mc_cpu_reduce.h                             \
	reduce/mc_cpu_reduce_int8.c                        \
	reduce/mc_cpu_reduce_int16.c                       \
	reduce/mc_cpu_reduce_int32.c                       \
	reduce/mc_cpu_reduce_int64.c                       \
	reduce/mc_cpu_reduce_uint8.c                       \
	reduce/mc_cpu_reduce_uint16.c                      \
	reduce/mc_cpu_reduce_uint32.c                      \
	reduce/mc_cpu_reduce_uint64.c                      \
	reduce/mc_cpu_reduce_float.c                       \
	reduce/mc_cpu_reduce_bfloat16.c                    \
	reduce/mc_cpu_reduce_double.c                      \
	reduce/mc_cpu_reduce_long_double.c                 \
	reduce/mc_cpu_reduce_float_complex.c               \
	reduce/mc_cpu_reduce_double_complex.c              \
	reduce/mc_cpu_reduce_long_double_complex.c         \
	reduce_alpha/mc_cpu_reduce_alpha_float.c           \
	reduce_alpha/mc_cpu_reduce_alpha_bfloat16.c        \
	reduce_alpha/mc_cpu_reduce_alpha_double.c          \
	reduce_alpha/mc_cpu_reduce_alpha_long.c            \
	reduce_alpha/mc_cpu_reduce_alpha_float_complex.c   \
	reduce_alpha/mc_cpu_reduce_alpha_double_complex.c  \
	reduce_alpha/mc_cpu_reduce_alpha_long_complex.c

module_LTLIBRARIES        = libucc_mc_cpu.la
libucc_mc_cpu_la_SOURCES  = $(sources)
libucc_mc_cpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_mc_cpu_la_CFLAGS   = $(BASE_CFLAGS)
libucc_mc_cpu_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed
libucc_mc_cpu_la_LIBADD   = $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
