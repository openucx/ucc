#!/bin/bash -eEl
progname=$(basename $0)
cov_exclude_file_list="test/gtest"

WS_URL=file://$WORKSPACE

function usage() 
{
	cat << HEREDOC

   Usage: $progname [--pre_script "./autogen.sh;./configure"] [--build_cmd "make all"] [--ignore_files "devx gtest"]  [--verbose]

   optional arguments:
     -h, --help           			show this help message and exit
     -p, --pre_script STRING        Preparation commands to run prior running coverity
     -b, --build_script STRING      Build command to pass to coverity
     -i, --ignore_files STRING  	Space separated list of files/dirs to ignore
     -v, --verbose        			increase the verbosity of the bash script

HEREDOC
exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--pre_script) pre_cmd="$2"; shift ;;
        -b|--build_script) build_cmd="$2"; shift ;;
        -i|--ignore_files) ignore_list="$2"; shift ;;
        -h|--help) usage ;;
        -v|--verbose) set +x ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
module load tools/cov

echo "Build with coverity"
cov_dir=${WORKSPACE}/jenkins/cov
cov_build_id="cov_build_${BUILD_NUMBER}"
cov_build="$cov_dir/$cov_build_id"
nproc=$(grep processor /proc/cpuinfo|wc -l)
make_opt="-j$(($nproc / 2 + 1))"
rm -rf $cov_build
set +eE
cov-build --dir $cov_build make $make_opt all
echo  "============ After Cov_build ==========="
set +eE
for excl in $cov_exclude_file_list; do
    cov-manage-emit --dir $cov_build --tu-pattern "file('$excl')" delete
done
set -eE
cov-analyze --dir $cov_build
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
    cat $cov_file  | grep -i -e '</\?TABLE\|</\?TD\|</\?TR\|</\?TH' | \
                     sed 's/^[\ \t]*//g' | tr -d '\n' | \
                     sed 's/<\/TR[^>]*>/\n/Ig'  | \
                     sed 's/<\/\?\(TABLE\|TR\)[^>]*>//Ig' | \
                     sed 's/^<T[DH][^>]*>\|<\/\?T[DH][^>]*>$//Ig' | \
                     sed 's/<\/T[DH][^>]*><T[DH][^>]*>/%/Ig' | \
                     cut -d"%" -f2,4,3 > $cov_build/index.csv

    filter_csv="$WORKSPACE/contrib/jenkins_tests/filter.csv"
    FILTER="grep -G -x -v -f $filter_csv $cov_build/index.csv"
    filtered_nerrors=$FILTER | wc -l
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