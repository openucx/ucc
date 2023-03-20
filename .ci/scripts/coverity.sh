#!/bin/bash -eE
#
# Copyright (c) 2001-2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
source $(dirname $0)/globals.sh

check_filter "Checking for coverity ..." "on"

cov_exclude_file_list="test/gtest"

if ! module_load tools/cov; then
	echo "WARNING: coverity is not found"
	exit 0
fi

echo "Build with coverity"
cov_dir=${WORKSPACE}/${prefix}/cov
cov_build_id="cov_build_${BUILD_NUMBER}"
cov_build="$cov_dir/$cov_build_id"
rm -rf $cov_build

set +eE
make $make_opt clean 2>&1 > /dev/null
module load $mpi_module
${WORKSPACE}/contrib/configure-devel --with-mpi=$OMPI_HOME --prefix=$sharp_dir -C
cov-build --dir $cov_build make $make_opt all

set +eE
for excl in $cov_exclude_file_list; do
    cov-manage-emit --dir $cov_build --tu-pattern "file('$excl')" delete
done
set -eE

cov-analyze --dir $cov_build
module unload $mpi_module
set -eE

cov_web_path="$(echo $cov_build | sed -e s,$WORKSPACE,,g)"
nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
rc=$(($rc+$nerrors))

index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
cov_url="$WS_URL/$cov_web_path/${index_html}"
cov_file="$cov_build/${index_html}"

filtered_nerrors=0
rm -f jenkins_sidelinks.txt
echo 1..1 > coverity.tap
if [ $nerrors -gt 0 ]; then

    #filter false positive

    #get  Table elements
    #     grep -i -e '</\?TABLE\|</\?TD\|</\?TR\|</\?TH'
    #Remove any Whitespace at the beginning of the line.
    #     sed 's/^[\ \t]*//g'
    #Remove newlines
    #     tr -d '\n\r'
    #Replace </TR> with newline
    #     sed 's/<\/TR[^>]*>/\n/Ig'
    #Remove TABLE and TR tags
    #     sed 's/<\/\?\(TABLE\|TR\)[^>]*>//Ig'
    #Remove ^<TD>, ^<TH>, </TD>$, </TH>$
    #     sed 's/^<T[DH][^>]*>\|<\/\?T[DH][^>]*>$//Ig'
    #Replace </TD><TD> with %
    #     sed 's/<\/T[DH][^>]*><T[DH][^>]*>/%/Ig'

    cat $cov_file  | grep -i -e '</\?TABLE\|</\?TD\|</\?TR\|</\?TH' | \
                     sed 's/^[\ \t]*//g' | tr -d '\n' | \
                     sed 's/<\/TR[^>]*>/\n/Ig'  | \
                     sed 's/<\/\?\(TABLE\|TR\)[^>]*>//Ig' | \
                     sed 's/^<T[DH][^>]*>\|<\/\?T[DH][^>]*>$//Ig' | \
                     sed 's/<\/T[DH][^>]*><T[DH][^>]*>/%/Ig' | \
                     cut -d"%" -f2,4,3 > $cov_build/index.csv

    filter_csv="$WORKSPACE/contrib/jenkins_tests/filter.csv"

    FILTER="grep -G -x -v -f $filter_csv $cov_build/index.csv"
    filtered_nerrors=`$FILTER | wc -l`

fi

if [ $filtered_nerrors -gt 0 ]; then
    echo "not ok 1 Coverity Detected $filtered_nerrors failures # $cov_url" >> coverity.tap
    info="Coverity found $filtered_nerrors errors"
    status="error"
else
    echo ok 1 Coverity found no issues >> coverity.tap
    info="Coverity found no issues"
    status="success"
fi

if [ -n "$ghprbGhRepository" ]; then
    context="MellanoxLab/coverity"
    do_github_status "repo='$ghprbGhRepository' sha1='$ghprbActualCommit' target_url='$cov_url' state='$status' info='$info' context='$context'"
fi

echo Coverity report: $cov_url
printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt

module unload tools/cov
