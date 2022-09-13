#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_TLS],[
    AC_ARG_WITH([tls],
        [AS_HELP_STRING([--with-tls=(tl1,tl2 | all)], [Enable build of TLs])],
        [TLS_REQUIRED=${with_tls}], [TLS_REQUIRED=^mlx5])
    AS_IF([test "x$with_docs_only" = xyes], TLS_REQUIRED="")
    m4_include([config/m4/tls_list.m4])
])

AC_DEFUN([CHECK_TLS_REQUIRED], [
    tl_name=$1
    AS_IF([ test "$TLS_REQUIRED" = "all" ], [CHECKED_TL_REQUIRED=y],
    [
       CHECKED_TL_REQUIRED=n
       required="y"
       tls_str=${TLS_REQUIRED}
       AS_IF([ test "${TLS_REQUIRED:0:1}" == "^" ], [
          CHECKED_TL_REQUIRED=y
          required="n"
          tls_str=${TLS_REQUIRED:1}])
       # AC_MSG_WARN([checking tl : $tl_name, TLS_REQUIRED=${TLS_REQUIRED}, tls_str=${tls_str}, required=${required}])
       for t in $(echo ${tls_str} | tr "," " "); do
           AS_IF([ test "$t" == "$tl_name" ], [CHECKED_TL_REQUIRED=$required], [])
       done
    ])
])
