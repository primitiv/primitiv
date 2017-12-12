#!/bin/bash
set -xe

# before_install
docker pull fedora:latest
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td fedora:latest /bin/bash

# install
docker exec travis-ci bash -c "dnf update -y"
docker exec travis-ci bash -c "dnf install -y rpm-build gcc-c++ cmake gtest-devel"

# install OpenCL environment
docker exec travis-ci bash -c "dnf install -y opencl-headers git hwloc-devel libtool-ltdl-devel ocl-icd-devel ocl-icd clang llvm-devel clang-devel zlib-devel blas-devel boost-devel patch --setopt=install_weak_deps=False"
docker exec travis-ci bash -c "git clone https://github.com/clMathLibraries/clBLAS.git"
docker exec travis-ci bash -c "cd ./clBLAS/src && cmake . -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TEST=OFF -DBUILD_KTEST=OFF"
docker exec travis-ci bash -c "cd ./clBLAS/src && make && make install"
# pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# We build the latest pocl instead of using distribution's package.
# See: https://github.com/pocl/pocl/issues/294
docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
docker exec travis-ci bash -c "cd ./pocl && make && make install"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_USE_OPENCL=ON -DPRIMITIV_BUILD_TESTS=ON"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"

# after_script
docker stop travis-ci
