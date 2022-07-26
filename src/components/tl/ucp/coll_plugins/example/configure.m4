# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# $COPYRIGHT$
# Additional copyrights may follow

CHECK_TLCP_REQUIRED("ucp_example")

AS_IF([test "$CHECKED_TLCP_REQUIRED" = "y"],
[
    tlcp_modules="${tlcp_modules}:ucp_example"
    tlcp_ucp_example_enabled=y
], [])

AM_CONDITIONAL([TLCP_UCP_EXAMPLE_ENABLED], [test "$tlcp_ucp_example_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/ucp/coll_plugins/example/Makefile])
