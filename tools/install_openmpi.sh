#!/bin/bash

module load python
module load cmake
module load rocm
module list

rocm_v="6.2.3"
ucx_v="1.16.x"
ucc_v="1.3.x"
ompi_v="5.0.x"

echo "ROCm version: ${rocm_v}"
echo "ROCM_PATH: ${ROCM_PATH}"

sleep 5

# Define install directories
INSTALL_DIR=/lustre/cursos/curso02/util
BUILD_DIR=${INSTALL_DIR}/build

export UCX_DIR=${INSTALL_DIR}/ucx
export UCC_DIR=${INSTALL_DIR}/ucc
export OMPI_DIR=${INSTALL_DIR}/ompi

mkdir -p ${BUILD_DIR}
mkdir -p ${UCX_DIR}
mkdir -p ${UCC_DIR}
mkdir -p ${OMPI_DIR}

if [ -d ${UCX_DIR}/lib ]; then
  echo "Found UCX in: ${UCX_DIR}"
else
  echo "Intalling UCX here: ${UCX_DIR}"
  cd ${BUILD_DIR}
  git clone --recursive -b v${ucx_v} https://github.com/openucx/ucx.git UCX
  cd UCX
  ./autogen.sh
  mkdir build; cd build
  ../contrib/configure-opt --prefix=$UCX_DIR --with-rocm=${ROCM_PATH} --without-knem --without-cuda 
  make -j 64
  make install
fi


if [ -d ${UCC_DIR}/lib ]; then
  echo "Found UCC in: ${UCC_DIR}"
else
  echo "Intalling UCC here: ${UCC_DIR}"
  cd ${BUILD_DIR}
  git clone --recursive -b v${ucc_v} https://github.com/openucx/ucc.git UCC
  cd UCC
  ./autogen.sh
  ./configure --with-rocm=${ROCM_PATH} --with-ucx=$UCX_DIR --prefix=$UCC_DIR
  make -j 64
  make install
fi

if [ -d ${OMPI_DIR}/lib ]; then
  echo "Found OMPI in: ${OMPI_DIR}"
else
  echo "Intalling OMPI here: ${OMPI_DIR}"
  cd ${BUILD_DIR}
  rm -rf OMPI
  git clone --recursive https://github.com/open-mpi/ompi.git OMPI
  cd OMPI
  git checkout -b v${ompi_v} 
  ./autogen.pl
  mkdir build; cd build
  ../configure --prefix=$OMPI_DIR --with-rocm=${ROCM_PATH} --with-ucx=$UCX_DIR --with-ucc=$UCC_DIR --without-verbs --with-libevent=internal LDFLAGS="-L/lib64"
  make -j 64
  make install
fi
