[![c++](https://img.shields.io/badge/c%2B%2B-11-blue.svg)](https://isocpp.org/)
[![backend](https://img.shields.io/badge/backend-CPU%2c%20CUDA%2c%20OpenCL-blue.svg)](README.md)
[![os](https://img.shields.io/badge/os-Ubuntu%2c%20Debian%2c%20Fedora%2c%20OSX-blue.svg)](https://travis-ci.org/odashi/primitiv)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build Status (master)](https://img.shields.io/travis/primitiv/primitiv/master.svg?label=build+%28master%29)](https://travis-ci.org/primitiv/primitiv)
[![Build Status (develop)](https://img.shields.io/travis/primitiv/primitiv/develop.svg?label=build+%28develop%29)](https://travis-ci.org/primitiv/primitiv)


primitiv
========

A Neural Network Toolkit.


Features
--------

- Dynamic and incremental graph construction (a.k.a. "define-by-run" style)
- On-demand memory allocation
- Automatic minibatch broadcasting
- Mostly device-independent
- Simple usage


Languages
---------

This repository contains only the core C++ library and its C APIs.
Some bindings for other programming languages (e.g., Python) can be found in [the official repository](https://github.com/primitiv).


Prerequisites
-------------

- CMake 3.1.0 or later
- C++11 compiler (GCC, Clang)
- (optional) [Google Test](https://github.com/google/googletest)
  - Required only when `-DPRIMITIV_BUILD_TESTS=ON`.
- (optional) CUDA 7.5 or later
  - Required only when `-DPRIMITIV_USE_CUDA=ON`
- (optional) OpenCL 1.2/OpenCL C++ binding v2
  - Required only when `-DPRIMITIV_USE_OPENCL=ON`


Install
-------

    git clone <this repository>
    cd primitiv
    mkdir build
    cd build
    cmake .. [-D(Options you need)]
    make [-j <threads>]
    [make test]
    [make install]


Build Options
-------------

- See [Build Options](doc/build_options.md).


Usage
-----

- See [Examples](examples).


Contact
-------

- [primitiv Developer Group](https://groups.google.com/forum/#!forum/primitiv-developer-group)
- [@odashi_t (maintainer) on Twitter](https://twitter.com/odashi_t)

This project is supported by [ASTREC](http://astrec.nict.go.jp/) in [NICT](http://nict.go.jp/).
