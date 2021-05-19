#!/bin/bash -eEx
set -o pipefail

# Install conda
#cd /tmp
#curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh -p /opt/conda -b
#rm -f Miniconda3-latest-Linux-x86_64.sh
#export PATH /opt/conda/bin:${PATH}

# Install conda python
#conda update -y conda
#conda install -c anaconda -y \
#    python \
#    pip \
#    scikit-learn
#pip3 install --no-cache-dir python-hostlist

#alternatives --set python /opt/conda/bin/python3
alternatives --set python /usr/bin/python3
pip3 install --user --upgrade setuptools wheel

command -v python
python --version

command -v python3
python3 --version

pip3 list
