#!/bin/bash -eEx
set -o pipefail

# UCC
echo "INFO: Install Torch-UCC (UCC version)"
export UCX_HOME=${UCX_INSTALL_DIR}
export UCC_HOME=${UCC_INSTALL_DIR}
export WITH_CUDA=${CUDA_HOME}
cd "${SRC_DIR}"
python setup.py install bdist_wheel
pip3 list | grep torch
python -c 'import torch, torch_ucc'
