#
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_CUDA

sources =     \
	mc_cuda_managed.h \
	mc_cuda_managed.c

module_LTLIBRARIES         = libucc_mc_cuda_managed.la
libucc_mc_cuda_managed_la_SOURCES  = $(sources)
libucc_mc_cuda_managed_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_mc_cuda_managed_la_CFLAGS   = $(BASE_CFLAGS)
libucc_mc_cuda_managed_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_mc_cuda_managed_la_LIBADD   = $(CUDA_LIBS)                      \
                                     $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
endif
