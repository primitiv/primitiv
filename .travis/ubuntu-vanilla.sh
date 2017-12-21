#!/bin/bash
set -xe

# install
sudo apt update
sudo apt install -y build-essential cmake3 libgtest-dev libeigen3-dev

# script
cd $TRAVIS_BUILD_DIR
cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/gtest
make VERBOSE=1
make test ARGS='-V'
make install
