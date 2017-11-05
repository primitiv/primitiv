#!/bin/bash
set -xe

# before_install
docker pull ubuntu:rolling
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td ubuntu:rolling /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y build-essential cmake libprotobuf-dev protobuf-compiler googletest python3-dev python3-pip python3-numpy"
docker exec travis-ci bash -c "pip3 install cython"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_BUILD_TESTS=ON -DGTEST_SOURCE_DIR=/usr/src/googletest/googletest"
docker exec travis-ci bash -c "cd /primitiv && make"
docker exec travis-ci bash -c "cd /primitiv && make test"
docker exec travis-ci bash -c "cd /primitiv && make install"
docker exec travis-ci bash -c "cd /primitiv/python-primitiv && ./setup.py build"
docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && cd /primitiv/python-primitiv && ./setup.py test"

# after_script
docker stop travis-ci
