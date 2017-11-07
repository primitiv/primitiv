#!/bin/bash
set -xe

# before_install
docker pull fedora:latest
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td fedora:latest /bin/bash

# install
docker exec travis-ci bash -c "yum update -y"
docker exec travis-ci bash -c "yum install -y gcc-c++ cmake git protobuf-devel protobuf-compiler gtest-devel python3-devel python3-numpy"
docker exec travis-ci bash -c "pip3 install cython"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_BUILD_TESTS=ON"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"
docker exec travis-ci bash -c "cd /primitiv/python-primitiv && ./setup.py build"
docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && cd /primitiv/python-primitiv && ./setup.py test"

# after_script
docker stop travis-ci
