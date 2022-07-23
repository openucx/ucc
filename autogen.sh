#!/bin/sh

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE:-$0}" )" &> /dev/null && pwd )"
if [ $SCRIPT_DIR != `pwd` ]; then
    echo "autogen.sh script must be launched from the top of the source tree"
    exit 1
fi

rm -rf config/m4/tl_coll_plugins_list.m4
rm -rf config/m4/tls_list.m4
rm -rf src/components/tl/makefile.am

touch config/m4/tl_coll_plugins_list.m4
touch config/m4/tls_list.m4

# Detect and generate TLs makefiles
for t in $(ls -d src/components/tl/*/); do
    echo "m4_include([$t/configure.m4])" >> config/m4/tls_list.m4
    plugin=$(basename $t)
    echo "SUBDIRS += components/tl/$plugin" >> src/components/tl/makefile.am
done

# Detect and generate TL coll plugins makefiles
for t in $(ls -d src/components/tl/*/); do
    if [ -d $t/coll_plugins ]; then
        rm -rf $t/makefile.coll_plugins.am
        for cp in $(ls -d $t/coll_plugins/*/); do
            echo "m4_include([$cp/configure.m4])" >> config/m4/tl_coll_plugins_list.m4
            plugin=$(basename $cp)
            echo "SUBDIRS += coll_plugins/$plugin" >> $t/makefile.coll_plugins.am
        done
    fi
done

rm -rf autom4te.cache
mkdir -p config/m4 config/aux
autoreconf -f -v --install || exit 1
rm -rf autom4te.cache

exit 0
