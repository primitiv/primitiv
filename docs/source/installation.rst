===================
Installing primitiv
===================


This document describes how to install primitiv to your computer.


Prerequisites
-------------


primitiv is designed based on a device-independent policy, and users can choose
dependencies between primitiv and other hardwares using
:doc:`build options </references/build_options>`.

For the minimal configuration (no other hardwares), primitiv requries below
softwares/libraries:

* C++11 compiler (`GCC <https://gcc.gnu.org/>`_, `Clang <https://clang.llvm.org/>`_, others)
* `CMake 3.1.0 <https://cmake.org/>`_

For building unit tests, it requires below libraries:

* `Google Test <https://github.com/google/googletest>`_

For using specific hardwares, it requires hardware-dependent libraries:

* ``devices::Eigen``
  * `Eigen 3.3.0 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_
* ``devices::CUDA``
  * `CUDA Toolkit 8.0 <https://developer.nvidia.com/cuda-toolkit>`_
  * `cuDNN 5.1.0 <https://developer.nvidia.com/cudnn>`_
* ``devices::OpenCL``
  * `OpenCL 1.2 <https://www.khronos.org/opencl/>`_
  * `OpenCL C++ Bindings 2 (cl2.hpp) <http://github.khronos.org/OpenCL-CLHPP/>`_
  * `clBLAS <https://github.com/clMathLibraries/clBLAS>`_


Installing from source
----------------------


blah
