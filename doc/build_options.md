*primitiv* Building Options
===========================


Standard Options
----------------

Users basically can use CMake 3.1.0 standard options
(e.g., `-DCMAKE_INSTALL_PREFIX`) together with the unique options.


Unique Options
--------------

<table>
  <tbody>
    <tr>
      <th>Name</th>
      <th>Default value</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>`PRIMITIV_BUILD_STATIC_LIBRARY`</td>
      <td>`OFF`</td>
      <td>Builds a static library instead of a shared object.</td>
    </tr>
    <tr>
      <td>`PRIMITIV_BUILD_TESTS`</td>
      <td>`OFF`</td>
      <td>
        Builds test binaries and generates `make test` command.
        This option introduces a dependency to the
        [Google Test Library](https://github.com/google/googletest), and
        [FindGTest](https://cmake.org/cmake/help/v3.1/module/FindGTest.html)
        options can also be used.
      </td>
    </tr>
      <td>`PRIMITIV_BUILD_TESTS_PROBABILISTIC`</td>
      <td>`OFF`</td>
      <td>Builds test cases that probabilistically fails.</td>
    </tr>
    <tr>
      <td>`PRIMITIV_GTEST_SOURCE_DIR`</td>
      <td>`""`</td>
      <td>
        Specifies the source directory of Google Test. If you want to use
        `googletest` module provided from Debian/Ubuntu repository, add
        `-DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest` together
        with `-PRIMITIV_BUILD_TESTS=ON` option.
      </td>
    </tr>
    <tr>
      <td>`PRIMITIV_USE_CACHE`</td>
      <td>`OFF`</td>
      <td>
        Whether or not to use cached values to prevent increasing computation
        amount.
        Libraries built with this flag will tend to consume more memory.
      </td>
    </tr>
    <tr>
      <td>`PRIMITIV_USE_CUDA`</td>
      <td>`OFF`</td>
      <td>
        Enables CUDA backend (`devices::CUDA` class).
        This option introduces a dependency to the
        [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), and
        [FindCUDA](https://cmake.org/cmake/help/v3.1/module/FindCUDA.html)
        options can also be used.
      </td>
    </tr>
    <tr>
      <td>`PRIMITIV_USE_OPENCL`</td>
      <td>`OFF`</td>
      <td>
        Enables OpenCL backend(`devices::OpenCL` class).
        This option introduces dependencies to a
        [OpenCL](https://www.khronos.org/opencl/) implementation and
        [OpenCL C++ Bindings v2](http://github.khronos.org/OpenCL-CLHPP/).
        [FindOpenCL](https://cmake.org/cmake/help/v3.1/module/FindOpenCL.html)
        options can also be used, and `cl2.hpp` should be found in
        `/path/to/include/CL`.
      </td>
    </tr>
  </tbody>
</table>
