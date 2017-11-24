[![c++](https://img.shields.io/badge/c%2B%2B-11-blue.svg)](https://isocpp.org/)
[![python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![backend](https://img.shields.io/badge/backend-CPU%2c%20CUDA%2c%20OpenCL-blue.svg)](README.md)
[![os](https://img.shields.io/badge/os-Ubuntu%2c%20Debian%2c%20Fedora%2c%20OSX-blue.svg)](https://travis-ci.org/odashi/primitiv)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build Status (master)](https://img.shields.io/travis/odashi/primitiv/master.svg?label=build+%28master%29)](https://travis-ci.org/odashi/primitiv)
[![Build Status (develop)](https://img.shields.io/travis/odashi/primitiv/develop.svg?label=build+%28develop%29)](https://travis-ci.org/odashi/primitiv)

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


Prerequisites
-------------

- CMake 3.1.0 or later
- C++11 compiler (GCC, Clang)
- (optional) [Google Test](https://github.com/google/googletest)
  - Required only when `-DPRIMITIV_BUILD_TESTS=ON`.
- (optional) CUDA 7.5 or later
  - Required only when `-DPRIMITIV_USE_CUDA=ON`

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

Building Options
----------------

- `PRIMITIV_BUILD_STATIC_LIBRARY` (default=`OFF`)
  - Builds a static library instead of a shared object.
- `PRIMITIV_BUILD_TESTS` (default=`OFF`)
  - Builds test binaries and generates `make test` command.
- `PRIMITIV_BUILD_TESTS_PROBABILISTIC` (default=`OFF`)
  - Builds test cases that probabilistically fails.
- `PRIMITIV_GTEST_SOURCE_DIR` (default=`""`)
  - Specifies the source directory of Google Test. If you want to use
    `googletest` module provided from Debian/Ubuntu repository,
    add `-DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest`
    together with `-PRIMITIV_BUILD_TESTS=ON` option.
- `PRIMITIV_USE_CACHE` (default=`OFF`)
  - Whether or not to use cached values to prevent increasing computation amount.
  - Libraries built with this flag will tend to consume more memory.
- `PRIMITIV_USE_CUDA` (default=`OFF`)
  - Enables CUDA backend (`devices::CUDA` class).
- `PRIMITIV_USE_OPENCL` (default=`OFF`)
  - Enables OpenCL backend(`devices::OpenCL` class).
- Other available options:
  - CMake standard options.
  - [FindGTest](https://cmake.org/cmake/help/v3.1/module/FindGTest.html) options.
  - [FindCUDA](https://cmake.org/cmake/help/v3.1/module/FindCUDA.html) options.
  - [FindOpenCL](https://cmake.org/cmake/help/v3.1/module/FindOpenCL.html) options.

Usage
-----

- [Short Python tutorial](https://github.com/odashi/primitiv/tree/develop/examples/tutorial1_xor.ipynb) with solving XOR problem.
- [Examples](https://github.com/odashi/primitiv/tree/develop/examples).


Contact
-------

- yus.takara at gmail.com
- [@odashi_t on Twitter](https://twitter.com/odashi_t)

This project is supported by [ASTREC](http://astrec.nict.go.jp/) in [NICT](http://nict.go.jp/).
