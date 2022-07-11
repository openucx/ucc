#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if TLCP_UCP_EXAMPLE_ENABLED
sources = example.c

module_LTLIBRARIES = libucc_tlcp_ucp_example.la
libucc_tlcp_ucp_example_la_SOURCES  = $(sources)
libucc_tlcp_ucp_example_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UCX_CPPFLAGS)
libucc_tlcp_ucp_example_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tlcp_ucp_example_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS)
libucc_tlcp_ucp_example_la_LIBADD   = $(UCX_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la \
                                      $(UCC_TOP_BUILDDIR)/src/components/tl/ucp/libucc_tl_ucp.la

include $(top_srcdir)/config/module.am

endif
