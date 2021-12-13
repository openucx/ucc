#!/bin/sh

rm -rf config/m4/tl_coll_plugins_list.m4
touch config/m4/tl_coll_plugins_list.m4
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
