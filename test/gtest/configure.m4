#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
# Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
#
# See file LICENSE for terms.
#

AC_LANG_PUSH([C++])

CHECK_COMPILER_FLAG([-fno-tree-vectorize], [-fno-tree-vectorize],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [GTEST_CXXFLAGS="$GTEST_CXXFLAGS -fno-tree-vectorize"],
                    [])

# error #236: controlling expression is constant
CHECK_COMPILER_FLAG([--diag_suppress 236], [--diag_suppress 236],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [GTEST_CXXFLAGS="$GTEST_CXXFLAGS --diag_suppress 236"],
                    [])

AC_LANG_POP([C++])

AC_SUBST([GTEST_CXXFLAGS], [$GTEST_CXXFLAGS])

AC_CONFIG_FILES([test/gtest/Makefile])
