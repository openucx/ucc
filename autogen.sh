#!/bin/sh
rm -rf autom4te.cache
mkdir -p config/m4 config/aux
autoreconf -f  --install || exit 1
rm -rf autom4te.cache
exit 0
