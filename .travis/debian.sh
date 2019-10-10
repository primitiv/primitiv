#!/bin/bash
set -xe

# before_install
docker pull debian:stable
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td debian:stable /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y build-essential cmake googletest libeigen3-dev"

# install OpenCL environment
docker exec travis-ci bash -c "apt install -y wget opencl-headers ocl-icd-dev ocl-icd-opencl-dev pocl-opencl-icd"
docker exec travis-ci bash -c "wget https://github.com/CNugteren/CLBlast/archive/1.2.0.tar.gz -O ./clblast.tar.gz"
docker exec travis-ci bash -c "mkdir ./clblast"
docker exec travis-ci bash -c "tar xf ./clblast.tar.gz -C ./clblast --strip-components 1"
docker exec travis-ci bash -c "cd ./clblast && cmake . && make && make install"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON -DPRIMITIV_BUILD_C_API=ON -DPRIMITIV_BUILD_TESTS=ON -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"

# after_script
docker stop travis-ci
