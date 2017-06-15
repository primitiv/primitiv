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

Build
-----

    mkdir build
    cd build
    cmake .. [-DUSE_CUDA=ON]
    make [-j <threads>]
    make test
