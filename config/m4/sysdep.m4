#
# Copyright (c) 2001-2014, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


AC_FUNC_ALLOCA


#
# SystemV shared memory
#
#IPC_INFO
AC_CHECK_LIB([rt], [shm_open],     [], AC_MSG_ERROR([librt not found]))
AC_CHECK_LIB([rt], [timer_create], [], AC_MSG_ERROR([librt not found]))


#
# Extended string functions
#
AC_CHECK_HEADERS([libgen.h])
AC_CHECK_DECLS([asprintf, basename, fmemopen], [],
				AC_MSG_ERROR([GNU string extensions not found]), 
				[#define _GNU_SOURCE 1
				 #include <string.h>
				 #include <stdio.h>
				 #ifdef HAVE_LIBGEN_H
				 #include <libgen.h>
				 #endif
				 ])



#
# Valgrind support
#
AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
                   [Enable Valgrind annotations (small runtime overhead, default NO)]),
    [],
    [with_valgrind=no]
)
AS_IF([test "x$with_valgrind" = xno],
      [AC_DEFINE([NVALGRIND], 1, [Define to 1 to disable Valgrind annotations.])],
      [AS_IF([test ! -d $with_valgrind], 
              [AC_MSG_NOTICE([Valgrind path was not defined, guessing ...])
               with_valgrind=/usr], [:])
        AC_CHECK_HEADER([$with_valgrind/include/valgrind/memcheck.h], [],
                       [AC_MSG_ERROR([Valgrind memcheck support requested, but <valgrind/memcheck.h> not found, install valgrind-devel rpm.])])
        CPPFLAGS="$CPPFLAGS -I$with_valgrind/include"
      ]
)
