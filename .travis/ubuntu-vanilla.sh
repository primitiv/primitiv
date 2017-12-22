#!/bin/bash
set -xe

# install
sudo apt update
sudo apt install -y build-essential cmake3 libgtest-dev

# install Eigen
#
# NOTE(vbkaisetsu):
# Ubuntu 14.04 contains Eigen 3.2
# primitiv requires 3.3 or later
#
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 -O ./eigen.tar.bz2
mkdir ./eigen
tar xf ./eigen.tar.bz2 -C ./eigen --strip-components 1
mkdir ./eigen/build
cd ./eigen/build
cmake ..
make && sudo make install

# script
cd $TRAVIS_BUILD_DIR
cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/gtest
make VERBOSE=1
make test ARGS='-V'
sudo make install
