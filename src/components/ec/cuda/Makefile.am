#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if HAVE_CUDA
SUBDIRS = kernel

sources =    \
	ec_cuda.h                          \
	ec_cuda.c                          \
	ec_cuda_executor.h                 \
	ec_cuda_executor.c                 \
	ec_cuda_executor_interruptible.c   \
	ec_cuda_executor_persistent.c      \
	ec_cuda_executor_persistent_wait.c \
	ec_cuda_resources.c                \
	ec_cuda_resources.h

module_LTLIBRARIES         = libucc_ec_cuda.la
libucc_ec_cuda_la_SOURCES  = $(sources)
libucc_ec_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_ec_cuda_la_CFLAGS   = $(BASE_CFLAGS)
libucc_ec_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_ec_cuda_la_LIBADD   = $(CUDA_LIBS)                      \
                             $(UCC_TOP_BUILDDIR)/src/libucc.la \
                             kernel/libucc_ec_cuda_kernels.la

include $(top_srcdir)/config/module.am
endif
