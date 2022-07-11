#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
#

if HAVE_ROCM
SUBDIRS = kernel

sources =                                \
	ec_rocm.h                        \
	ec_rocm.c                        \
	ec_rocm_executor.h               \
	ec_rocm_executor.c               \
	ec_rocm_executor_interruptible.c \
	ec_rocm_executor_persistent.c

module_LTLIBRARIES         = libucc_ec_rocm.la
libucc_ec_rocm_la_SOURCES  = $(sources)
libucc_ec_rocm_la_CPPFLAGS = $(AM_CPPFLAGS) $(HIP_CPPFLAGS) $(ROCM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_ec_rocm_la_CFLAGS   = $(BASE_CFLAGS)
libucc_ec_rocm_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(HIP_LDFLAGS) $(ROCM_LDFLAGS)
libucc_ec_rocm_la_LIBADD   = $(HIP_LIBS) $(ROCM_LIBS)          \
			     kernel/libucc_ec_rocm_kernels.la  \
			     $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
endif
