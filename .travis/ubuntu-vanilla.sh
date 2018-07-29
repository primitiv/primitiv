#!/bin/bash
set -xe

# install
sudo apt update
sudo apt install -y build-essential cmake3 libgtest-dev

# download Eigen
#
# NOTE(vbkaisetsu):
# Ubuntu 14.04 contains Eigen 3.2
# primitiv requires 3.3 or later
#
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 -O ./eigen.tar.bz2
mkdir $TRAVIS_BUILD_DIR/eigen
tar xf ./eigen.tar.bz2 -C $TRAVIS_BUILD_DIR/eigen --strip-components 1

# script
cd $TRAVIS_BUILD_DIR
cmake . -DPRIMITIV_USE_EIGEN=ON -DEIGEN3_INCLUDE_DIR=$TRAVIS_BUILD_DIR/eigen -DPRIMITIV_BUILD_C_API=ON -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/gtest
make VERBOSE=1
make test ARGS='-V'
sudo make install
