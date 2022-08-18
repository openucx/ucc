#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
#
if TL_RCCL_ENABLED

allgatherv =                      \
	allgatherv/allgatherv.h       \
	allgatherv/allgatherv.c

sources =             \
	tl_rccl.h         \
	tl_rccl.c         \
	tl_rccl_lib.c     \
	tl_rccl_context.c \
	tl_rccl_team.c    \
	tl_rccl_coll.h    \
	tl_rccl_coll.c    \
	$(allgatherv)

module_LTLIBRARIES = libucc_tl_rccl.la
libucc_tl_rccl_la_SOURCES  = $(sources)
libucc_tl_rccl_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS) $(HIP_CPPFLAGS) $(RCCL_CPPFLAGS)
libucc_tl_rccl_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_rccl_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(ROCM_LDFLAGS) $(HIP_LDFLAGS) $(RCCL_LDFLAGS)
libucc_tl_rccl_la_LIBADD   = $(ROCM_LIBS) $(HIP_LIBS) $(RCCL_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
