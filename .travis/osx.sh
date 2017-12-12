#!/bin/bash
set -xe

# install
brew update
git clone https://github.com/google/googletest.git $TRAVIS_BUILD_DIR/googletest

# script
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH::$TRAVIS_BUILD_DIR/googletest/googletest/include
cd $TRAVIS_BUILD_DIR
cmake . -DPRIMITIV_BUILD_TESTS=ON -DPRIMITIV_GTEST_SOURCE_DIR=$TRAVIS_BUILD_DIR/googletest/googletest
make VERBOSE=1
make test ARGS='-V'
make install
