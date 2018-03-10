=============
Build Options
=============


Standard Options
----------------


Users basically can use *CMake 3.1.0* standard options
(e.g., ``-DCMAKE_INSTALL_PREFIX``) together with the unique options.


Unique Options
--------------


PRIMITIV_BUILD_C_API
    Default value: ``OFF``

    Builds C APIs.
    ``libprimitiv_c`` library file and headers in the ``primitiv/c`` directory
    will also be installed.

PRIMITIV_BUILD_STATIC_LIBRARY
    Default value: ``OFF``

    Builds static libraries instead of shared objects.

PRIMITIV_BUILD_TESTS
    Default value: ``OFF``

    Builds test binaries and generates ``make test`` command.
    This option introduces a dependency to the
    `Google Test <https://github.com/google/googletest>`_.
    `FindGTest <https://cmake.org/cmake/help/v3.1/module/FindGTest.html>`_
    options can also be used.

PRIMITIV_BUILD_TESTS_PROBABILISTIC
    Default value: ``OFF``

    Builds test cases that probabilistically fails.

PRIMITIV_GTEST_SOURCE_DIR
    Default value: ``""``

    Specifies the source directory of Google Test. If you want to use Google
    Test provided from Debian/Ubuntu repository, add
    ``-DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest``
    together with ``-PRIMITIV_BUILD_TESTS=ON`` option.

PRIMITIV_USE_CACHE
    Default value: ``OFF``

    Whether or not to use cached values to prevent increasing computation
    amount.
    Libraries built with this flag will tend to consume more memory.

PRIMITIV_USE_EIGEN
    Default value: ``OFF``

    Enables Eigen backend (``primitiv::devices::Eigen`` class).
    This option introduces a dependency to the
    `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_
    library, and
    `FindEigen3 </cmake/FindEigen3.cmake>`_
    options can also be used.

PRIMITIV_USE_CUDA
    Default value: ``OFF``

    Enables CUDA backend (``primitiv::devices::CUDA`` class).
    This option introduces a dependency to the
    `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_
    v8.0 or later.
    `FindCuda <https://cmake.org/cmake/help/v3.1/module/FindCUDA.html>`_
    options can also be used.

PRIMITIV_USE_CUDNN
    Default value: ``OFF``

    Enables cuDNN as the backend of few CUDA functions.
    This option introduces a dependency to the
    `cuDNN library <https://developer.nvidia.com/cudnn>`_
    v5.0 or later.
    `FindCuDNN </cmake/FindCuDNN.cmake>`_
    options can also be used.

PRIMITIV_USE_OPENCL
    Default value: ``OFF``

    Enables OpenCL backend(``primitiv::devices::OpenCL`` class).
    This option introduces dependencies to an
    `OpenCL <https://www.khronos.org/opencl/>`_
    v1.2 implementation and
    `OpenCL C++ Bindings <http://github.khronos.org/OpenCL-CLHPP/>`_
    v2.
    `FindOpenCL <https://cmake.org/cmake/help/v3.1/module/FindOpenCL.html>`_
    options can also be used, and ``cl2.hpp`` should be found in
    ``/path/to/include/CL``.
