#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
if TL_NCCL_ENABLED

allgatherv =                      \
	allgatherv/allgatherv.h       \
	allgatherv/allgatherv.c

sources =             \
	tl_nccl.h         \
	tl_nccl.c         \
	tl_nccl_lib.c     \
	tl_nccl_context.c \
	tl_nccl_team.c    \
	tl_nccl_coll.h    \
	tl_nccl_coll.c    \
	$(allgatherv)

module_LTLIBRARIES = libucc_tl_nccl.la
libucc_tl_nccl_la_SOURCES  = $(sources)
libucc_tl_nccl_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS) $(NCCL_CPPFLAGS)
libucc_tl_nccl_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_nccl_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS) $(NCCL_LDFLAGS)
libucc_tl_nccl_la_LIBADD   = $(CUDA_LIBS) $(NCCL_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am

endif
