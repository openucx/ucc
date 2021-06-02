#!/bin/bash -eEx
set -o pipefail

command -v mpirun
export UCX_WARN_UNUSED_ENV_VARS=n
ucx_info -e -u t

#==============================================================================
# CPU
#==============================================================================
echo "INFO: UCC barrier (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_barrier_test.py --backend=gloo

echo "INFO: UCC alltoall (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_alltoall_test.py --backend=gloo

echo "INFO: UCC alltoallv (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_alltoallv_test.py --backend=gloo

echo "INFO: UCC allgather (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_allgather_test.py --backend=gloo

echo "INFO: UCC allreduce (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_allreduce_test.py --backend=gloo

echo "INFO: UCC broadcast (CPU)"
/bin/bash ${SRC_DIR}/test/start_test.sh ${SRC_DIR}/test/torch_bcast_test.py --backend=gloo
