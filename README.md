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

- CMake 3.1.0 or later
- GCC 4.9 or later (maybe 4.8 is OK)
- (optional) CUDA 8.0 or later


Dependencies
------------

Following libraries will be included into the build tree by running
`git submodule init/update`.

- [Google Test](https://github.com/google/googletest)
  - Required only when `-DPRIMITIV_BUILD_TESTS=ON`.
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

*Attention*: `make install` will attempt to install also above libraries into
your install prefix.

Build
-----

    git clone <this repository>
    cd primitiv
    git submodule init
    git submodule update
    mkdir build
    cd build
    cmake .. [-DPRIMITIV_USE_CUDA=ON] [(Other options listed below if necessary)]
    make [-j <threads>]
    [make test]
    [make install]

Building Options
----------------

- `PRIMITIV_BUILD_STATIC_LIBRARY` (default=`OFF`)
  - Builds a static library instead of a shared object.
- `PRIMITIV_BUILD_TESTS` (default=`ON`)
  - Builds test binaries and generates `make test` command.
- `PRIMITIV_USE_CACHE` (default=`OFF`)
  - Whether or not to use cached values to prevent increasing computation amount.
  - Libraries built with this flag will tend to consume more memory.
- `PRIMITIV_USE_CUDA` (default=`OFF`)
  - Enables CUDA backend (`CUDADevice` class).
- `GOOGLETEST_INCLUDE_DIR`, `GOOGLETEST_LIBRARY_DIR`
  - Optional directories which stores [Google Test](https://github.com/google/googletest) include files & libraries.
- Other CMake standard options and [FindCUDA](https://cmake.org/cmake/help/v3.0/module/FindCUDA.html) options are available.

Usage
-----

See [examples](https://github.com/odashi/primitiv/tree/master/example).


Contact
-------

- yus.takara at gmail.com
- [@odashi_t on Twitter](https://twitter.com/odashi_t)

This project is supported by [ASTREC](http://astrec.nict.go.jp/) in [NICT](http://nict.go.jp/).
