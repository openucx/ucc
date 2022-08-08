#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_CUDA

sources =     \
	mc_cuda.h \
	mc_cuda.c

module_LTLIBRARIES         = libucc_mc_cuda.la
libucc_mc_cuda_la_SOURCES  = $(sources)
libucc_mc_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_mc_cuda_la_CFLAGS   = $(BASE_CFLAGS)
libucc_mc_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_mc_cuda_la_LIBADD   = $(CUDA_LIBS)                      \
                             $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
endif
