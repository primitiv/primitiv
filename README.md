primitiv
========

A Neural Network Toolkit.

Features
--------

- Dynamic and incremental graph construction
- On-demand memory allocation
- Automatic minibatch broadcasting
- Mostly device-independent
- Simple usage

Prerequisites
-------------

- CMake 3.1.0
- GCC 4.9
- (optional) CUDA 8.0

Build
-----

    git clone <this repository>
    cd primitiv
    git submodule init
    git submodule update
    mkdir build
    cd build
    cmake .. [-DUSE_CUDA=ON]
    make [-j <threads>]
    make test
