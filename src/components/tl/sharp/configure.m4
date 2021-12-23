#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

tl_sharp_enabled=n
CHECK_TLS_REQUIRED(["sharp"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    CHECK_SHARP
    AC_MSG_RESULT([SHARP support: $sharp_happy])
    if test $sharp_happy = "yes"; then
       tl_modules="${tl_modules}:sharp"
       tl_sharp_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_SHARP_ENABLED], [test "$tl_sharp_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/sharp/Makefile])
