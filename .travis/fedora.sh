#!/bin/bash
set -xe

# before_install
docker pull fedora:latest
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td fedora:latest /bin/bash

# install
docker exec travis-ci bash -c "dnf update -y"
docker exec travis-ci bash -c "dnf install -y rpm-build gcc-c++ cmake make gtest-devel eigen3-devel"

# install OpenCL environment
docker exec travis-ci bash -c "dnf install -y opencl-headers git wget hwloc-devel libtool-ltdl-devel ocl-icd-devel ocl-icd clang llvm-devel clang-devel zlib-devel --setopt=install_weak_deps=False"
docker exec travis-ci bash -c "wget https://github.com/CNugteren/CLBlast/archive/1.2.0.tar.gz -O ./clblast.tar.gz"
docker exec travis-ci bash -c "mkdir ./clblast"
docker exec travis-ci bash -c "tar xf ./clblast.tar.gz -C ./clblast --strip-components 1"
docker exec travis-ci bash -c "cd ./clblast && cmake . && make && make install"
# pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# We build the latest pocl instead of using distribution's package.
# See: https://github.com/pocl/pocl/issues/294
docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
docker exec travis-ci bash -c "cd ./pocl && make && make install"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON -DPRIMITIV_BUILD_C_API=ON -DPRIMITIV_BUILD_TESTS=ON"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"

# after_script
docker stop travis-ci
