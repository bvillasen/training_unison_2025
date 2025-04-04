#!/bin/bash

module load python
module load cmake
module load rocm
module list

export INSTALL_DIR=$PWD
export INSTALL_PATH=$INSTALL_DIR/rocprof-compute
export BUILD_DIR=$INSTALL_DIR/build

mkdir -p $BUILD_DIR

cd ${BUILD_DIR}
git clone https://github.com/ROCm/rocprofiler-compute

cd rocprofiler-compute
# python -m pip install -t ${INSTALL_PATH}/python-libs -r requirements.txt --upgrade
# pip install -t ${INSTALL_PATH}/python-libs -r requirements.txt --upgrade
pip install -r requirements.txt --upgrade

      # -DMOD_INSTALL_PATH=${INSTALL_PATH}/modulefiles \
      # -DPYTHON_DEPS=${INSTALL_PATH}/python-libs \

mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}/ \
      -DCMAKE_BUILD_TYPE=Release \
      ..

make install
