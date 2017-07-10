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
    cmake ..
        [-DPRIMITIV_BUILD_STATIC_LIBRARY=ON]
        [-DPRIMITIV_BUILD_TESTS=ON]
        [-DPRIMITIV_USE_CACHE=ON]
        [-DPRIMITIV_USE_CUDA=ON]
        [Other CMake/CMakeCUDA options if necessary]
    make [-j <threads>]
    [make test]
    [make install]


Usage
-----

See [examples](https://github.com/odashi/primitiv/tree/master/example).


Contact
-------

- yus.takara at gmail.com
- [@odashi_t on Twitter](https://twitter.com/odashi_t)

This project is supported by [ASTREC](http://astrec.nict.go.jp/) in [NICT](http://nict.go.jp/).
