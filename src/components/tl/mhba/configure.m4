#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

tl_mhba_enabled=n
CHECK_TLS_REQUIRED(["mhba"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    # CHECK_MHBA
    mhba_happy=yes
    AC_MSG_RESULT([MHBA support: $mhba_happy])
    if test $mhba_happy = "yes"; then
        tl_modules="${tl_modules}:mhba"
        tl_mhba_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_MHBA_ENABLED], [test "$tl_mhba_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/mhba/Makefile])
