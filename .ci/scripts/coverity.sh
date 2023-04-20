#!/bin/bash 

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
nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
rc=$(($rc+$nerrors))

index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
cov_url="$WS_URL/$cov_web_path/${index_html}"





echo Coverity report: $cov_url
printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt

module unload tools/cov


exit $rc
