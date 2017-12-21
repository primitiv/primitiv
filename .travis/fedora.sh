#!/bin/bash
set -xe

# before_install
docker pull fedora:latest
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td fedora:latest /bin/bash

# install
docker exec travis-ci bash -c "dnf update -y"
docker exec travis-ci bash -c "dnf install -y rpm-build gcc-c++ cmake gtest-devel"

# install Eigen
#
# NOTE(vbkaisetsu):
# Fedora 27 contains Eigen 3.3.4 and gcc 7.2.1.
# gcc/g++ 7 detects int-in-bool-context error in the latest released version of Eigen
# by default, and it will be fixed in 3.3.5. For now, this script downloads the latest
# development version to solve this problem.
#
# For more details, see: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1402
#
docker exec travis-ci bash -c "dnf install -y mercurial"
docker exec travis-ci bash -c "hg clone https://bitbucket.org/eigen/eigen"
docker exec travis-ci bash -c "mkdir ./eigen/build"
docker exec travis-ci bash -c "cd ./eigen/build && cmake .."
docker exec travis-ci bash -c "cd ./eigen/build && make && make install"

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
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON -DPRIMITIV_BUILD_TESTS=ON"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"

# after_script
docker stop travis-ci
