#!/bin/bash -eEl
progname=$(basename $0)

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

topdir=$(git rev-parse --show-toplevel)
cd $topdir


if [ ! -d .git ]; then
	echo "Error: should be run from project root"
	exit 1
fi


ncpus=$(cat /proc/cpuinfo|grep processor|wc -l)
export AUTOMAKE_JOBS=$ncpus

if [ -n "${pre_cmd}" ]; then

    echo "==== Running Pre-commands ===="

    set +eE
    /bin/bash -c "$pre_cmd"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo pre-commands failed
        exit 1
    fi

    set -eE
fi

cov_build="cov_build"
rm -rf $cov_build

module load tools/cov

echo "==== Running coverity ===="

cov-build --dir $cov_build $build_cmd all

if [ -n "${ignore_list}" ]; then

    echo "==== Adding ignore list ===="

    for item in ${ignore_list}; do
        cov-manage-emit --dir ${cov_build} --tu-pattern "file(${item})" delete ||:
    done
fi

echo "==== Running anaysis ===="

cov-analyze --jobs $ncpus $COV_OPT --security --concurrency --dir $cov_build
cov-format-errors --dir $cov_build --emacs-style |& tee cov_${variant}.log

nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
rc=$(($rc+$nerrors))

echo status $rc

exit $rc
