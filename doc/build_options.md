*primitiv* Build Options
=========================


Standard Options
----------------

Users basically can use CMake 3.1.0 standard options
(e.g., `-DCMAKE_INSTALL_PREFIX`) together with the unique options.


Unique Options
--------------

<table>
  <tbody>
    <tr>
    <tr>
      <th>Name</th>
      <th>Default value</th>
      <th>Description</th>
    </tr>
      <td><code>PRIMITIV_BUILD_C_API</code></td>
      <td><code>OFF</code></td>
      <td>
        Builds C APIs.
        <code>libprimitiv_c</code> library file and headers in the
        <code>primitiv/c</code> directory will also be installed.
      </td>
    </tr>
    <tr>
      <td><code>PRIMITIV_BUILD_STATIC_LIBRARY</code></td>
      <td><code>OFF</code></td>
      <td>Builds static libraries instead of shared objects.</td>
    </tr>
    <tr>
      <td><code>PRIMITIV_BUILD_TESTS</code></td>
      <td><code>OFF</code></td>
      <td>
        Builds test binaries and generates <code>make test</code> command.
        This option introduces a dependency to the
        <a href="https://github.com/google/googletest">Google Test</a>, and
        <a href="https://cmake.org/cmake/help/v3.1/module/FindGTest.html">FindGTest</a>
        options can also be used.
      </td>
    </tr>
      <td><code>PRIMITIV_BUILD_TESTS_PROBABILISTIC</code></td>
      <td><code>OFF</code></td>
      <td>Builds test cases that probabilistically fails.</td>
    </tr>
    <tr>
      <td><code>PRIMITIV_GTEST_SOURCE_DIR</code></td>
      <td><code>""</code></td>
      <td>
        Specifies the source directory of Google Test. If you want to use
        Google Test provided from Debian/Ubuntu repository, add
        <code>-DPRIMITIV_GTEST_SOURCE_DIR=/usr/src/googletest/googletest</code>
        together with <code>-PRIMITIV_BUILD_TESTS=ON</code> option.
      </td>
    </tr>
    <tr>
      <td><code>PRIMITIV_USE_CACHE</code></td>
      <td><code>OFF</code></td>
      <td>
        Whether or not to use cached values to prevent increasing computation
        amount.
        Libraries built with this flag will tend to consume more memory.
      </td>
    </tr>
    <tr>
      <td><code>PRIMITIV_USE_EIGEN</code></td>
      <td><code>OFF</code></td>
      <td>
        Enables Eigen backend (<code>primitiv::devices::Eigen</code> class).
        This option introduces a dependency to the
        <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page">Eigen3 library</a>,
        and
        <a href="/cmake/FindEigen3.cmake">FindEigen3</a>
        options can also be used.
      </td>
    </tr>
    <tr>
      <td><code>PRIMITIV_USE_CUDA</code></td>
      <td><code>OFF</code></td>
      <td>
        Enables CUDA backend (<code>primitiv::devices::CUDA</code> class).
        This option introduces a dependency to the
        <a href="https://developer.nvidia.com/cuda-toolkit">NVIDIA CUDA Toolkit v7.5 or later</a>,
        and
        <a href="https://cmake.org/cmake/help/v3.1/module/FindCUDA.html">FindCUDA</a>
        options can also be used.
      </td>
    </tr>
    <tr>
      <td><code>PRIMITIV_USE_OPENCL</code></td>
      <td><code>OFF</code></td>
      <td>
        Enables OpenCL backend(<code>primitiv::devices::OpenCL</code> class).
        This option introduces dependencies to an
        <a href="https://www.khronos.org/opencl/">OpenCL v1.2</a> implementation and
        <a href="http://github.khronos.org/OpenCL-CLHPP/">OpenCL C++ Bindings v2</a>.
        <a href="https://cmake.org/cmake/help/v3.1/module/FindOpenCL.html">FindOpenCL</a>
        options can also be used, and <code>cl2.hpp</code> should be found in
        <code>/path/to/include/CL</code>.
      </td>
    </tr>
  </tbody>
</table>
