# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_GTEST], [
dnl Provide a flag to enable or disable Google Test usage.
    AC_ARG_ENABLE([gtest], [AS_HELP_STRING([--enable-gtest],
                  [Enable tests using the Google C++ Testing Framework.
                  (Default is disabled.)])],
                  [enable_gtest=$enableval],
                  [enable_gtest=no])

    AM_CONDITIONAL([HAVE_GTEST],[test "x$enable_gtest" = "xyes"])
    AS_IF([test "x$enable_gtest" == "xyes"],
          [gtest_enable=enabled], [gtest_enable=disabled])
])
