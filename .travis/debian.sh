#!/bin/bash
set -xe

# before_install
docker pull debian:stable
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv -td debian:stable /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y build-essential cmake googletest python3-dev python3-pip python3-numpy"
docker exec travis-ci bash -c "pip3 install cython"

# script
docker exec travis-ci bash -c "cd /primitiv && cmake . -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest"
docker exec travis-ci bash -c "cd /primitiv && make VERBOSE=1"
docker exec travis-ci bash -c "cd /primitiv && make test ARGS='-V'"
docker exec travis-ci bash -c "cd /primitiv && make install"
docker exec travis-ci bash -c "cd /primitiv/python-primitiv && ./setup.py build"
docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && cd /primitiv/python-primitiv && ./setup.py test"

# after_script
docker stop travis-ci
