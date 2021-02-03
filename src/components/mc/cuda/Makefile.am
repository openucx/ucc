#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

if HAVE_CUDA
sources =    \
	mc_cuda.h \
	mc_cuda.c

module_LTLIBRARIES         = libucc_mc_cuda.la
libucc_mc_cuda_la_SOURCES  = $(sources)
libucc_mc_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_mc_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_mc_cuda_la_LIBADD   = $(CUDA_LIBS)

include $(top_srcdir)/config/module.am
endif
