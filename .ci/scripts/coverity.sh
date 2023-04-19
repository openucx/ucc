#!/bin/bash -eEl

topdir=$(git rev-parse --show-toplevel)
cd $topdir


if [ ! -d .git ]; then
	echo "Error: should be run from project root"
	exit 1
fi

echo "==== Running coverity ===="

ncpus=$(cat /proc/cpuinfo|grep processor|wc -l)
export AUTOMAKE_JOBS=$ncpus

./autogen.sh
./configure 
make -j $ncpus clean

cov_build="cov_build"
rm -rf $cov_build

module load tools/cov

cov-build --dir $cov_build make -j $ncpus 
cov-analyze --jobs $ncpus $COV_OPT --security --concurrency --dir $cov_build
cov-format-errors --dir $cov_build --emacs-style |& tee cov.log

cov_web_path="$(echo $cov_build | sed -e s,$WORKSPACE,,g)"

nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
rc=$(($rc+$nerrors))

index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
cov_url="$WS_URL/$cov_web_path/${index_html}"
cov_file="$cov_build/${index_html}"
if [ $nerrors -gt 0 ]; then
	cat $cov_file  | grep -i -e '</\?TABLE\|</\?TD\|</\?TR\|</\?TH' | \
                     sed 's/^[\ \t]*//g' | tr -d '\n' | \
                     sed 's/<\/TR[^>]*>/\n/Ig'  | \
                     sed 's/<\/\?\(TABLE\|TR\)[^>]*>//Ig' | \
                     sed 's/^<T[DH][^>]*>\|<\/\?T[DH][^>]*>$//Ig' | \
                     sed 's/<\/T[DH][^>]*><T[DH][^>]*>/%/Ig' | \
                     cut -d"%" -f2,4,3 > $cov_build/index.csv
	FILTER="grep -G -x -v -f $cov_build/index.csv"
	filtered_nerrors="$FILTER | wc -l"

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

echo Coverity report: $cov_url
printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt

module unload tools/cov


exit $rc
