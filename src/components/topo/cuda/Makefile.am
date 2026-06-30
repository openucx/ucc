#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if HAVE_CUDA

sources =        \
	ucc_sysinfo_cuda.h \
	ucc_sysinfo_cuda.c

module_LTLIBRARIES            = libucc_sysinfo_cuda.la
libucc_sysinfo_cuda_la_SOURCES  = $(sources)
libucc_sysinfo_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_sysinfo_cuda_la_CFLAGS   = $(BASE_CFLAGS)
libucc_sysinfo_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_sysinfo_cuda_la_LIBADD   = $(CUDA_LIBS) $(NVML_LIBS)               \
                                  $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
endif
