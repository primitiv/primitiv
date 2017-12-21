#!/bin/bash
set -xe

# before_install
docker pull ubuntu:rolling
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td ubuntu:rolling /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y build-essential cmake googletest"

# install Eigen
#
# NOTE(vbkaisetsu):
# Ubuntu 17.04  contains Eigen 3.3.4 and gcc 7.2.0.
# gcc/g++ 7 detects int-in-bool-context error in the latest released version of Eigen
# by default, and it will be fixed in 3.3.5. For now, this script downloads the latest
# development version to solve this problem.
#
# For more details, see: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1402
#
docker exec travis-ci bash -c "apt install -y mercurial"
docker exec travis-ci bash -c "hg clone https://bitbucket.org/eigen/eigen"
docker exec travis-ci bash -c "mkdir ./eigen/build"
docker exec travis-ci bash -c "cd ./eigen/build && cmake .."
docker exec travis-ci bash -c "cd ./eigen/build && make && make install"

# install OpenCL environment
docker exec travis-ci bash -c "apt install -y opencl-headers libclblas-dev git pkg-config libhwloc-dev libltdl-dev ocl-icd-dev ocl-icd-opencl-dev clang-3.8 llvm-3.8-dev libclang-3.8-dev libz-dev"
# pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# We build the latest pocl instead of using distribution's package.
# See: https://github.com/pocl/pocl/issues/294
docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
docker exec travis-ci bash -c "cd ./pocl && make && make install"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"

# after_script
docker stop travis-ci
