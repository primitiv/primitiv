===================
Installing primitiv
===================


This section describes how to install primitiv to your computer.


Prerequisites
-------------


primitiv is designed based on a device-independent policy, and you can choose
dependencies between primitiv and other hardwares using
:doc:`build options </reference/build_options>`.

For the minimal configuration (no other hardwares), primitiv requries below
softwares/libraries:

* C++11 compiler (`GCC <https://gcc.gnu.org/>`_, `Clang <https://clang.llvm.org/>`_, others)
* `CMake 3.1.0 <https://cmake.org/>`_ or later

For building unit tests, it requires below libraries:

* `Google Test <https://github.com/google/googletest>`_

For using specific hardwares, it requires some hardware-dependent libraries:

* ``primitiv::devices::Eigen``

  * `Eigen 3.3.0 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ or later

* ``primitiv::devices::CUDA``

  * `CUDA Toolkit 8.0 <https://developer.nvidia.com/cuda-toolkit>`_ or later
  * `cuDNN 5.1.0 <https://developer.nvidia.com/cudnn>`_ or later

* ``primitiv::devices::OpenCL``

  * `OpenCL 1.2 <https://www.khronos.org/opencl/>`_
  * `OpenCL C++ Bindings 2 (cl2.hpp) <http://github.khronos.org/OpenCL-CLHPP/>`_
  * `CLBlast 1.2.0 <https://github.com/CNugteren/CLBlast>`_


Installing primitiv from source (Debian/Ubuntu)
-----------------------------------------------


Installing common prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: shell
  
  $ apt install build-essential cmake


Installing Eigen
^^^^^^^^^^^^^^^^


Although you can install primitiv without any specific hardwares, we recommend
to bind at least the **Eigen** backend to compute your neural networks much
faster on CPUs.


.. code-block:: shell

  $ apt install wget
  $ cd /path/to/your/src
  $ mkdir -p eigen
  $ wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
  $ tar -xf 3.3.4.tar.bz2 -C eigen --strip-components=1
  $ rm 3.3.4.tar.bz2


Installing primitiv
^^^^^^^^^^^^^^^^^^^


To select primitiv versions to be installed, you can retrieve some archives from
`official releases <https://github.com/primitiv/primitiv/releases>`_.

.. code-block:: shell

  $ cd /path/to/your/src
  $ mkdir -p primitiv
  $ wget https://github.com/primitiv/primitiv/archive/v0.3.1.tar.gz
  $ tar -xf v0.3.1.tar.gz -C primitiv --strip-components=1
  $ rm v0.3.1.tar.gz

Also, you can download a development (or other specific) branch using Git:

.. code-block:: shell

  $ ce /path/to/your/src
  $ apt install git
  $ git clone https://github.com/primitiv/primitiv -b develop

Then we build primitiv using a standard process of CMake:

.. code-block:: shell

  $ cd /path/to/your/src/primitiv
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make
  $ make install

``make install`` will create ``libprimitiv.so`` in the system library directory
and ``primitiv`` directory in the system include directory.

In some cases, you also need to add the path to the library directory to the
``${LD_LIBRARY_PATH}`` environment variable:

.. code-block:: shell

  $ export LD_LIBRARY_PATH=/path/to/your/lib:${LD_LIBRARY_PATH}

If we use the Eigen backend, specify both ``EIGEN3_INCLUDE_DIR`` and
``PRIMITIV_USE_EIGEN`` options to ``cmake``:

.. code-block:: shell

  $ cmake .. \
    -DEIGEN3_INCLUDE_DIR=/path/to/your/src/eigen \
    -DPRIMITIV_USE_EIGEN=ON


Installing primitiv with CUDA
-----------------------------


.. code-block:: shell

  $ cmake .. -DPRIMITIV_USE_CUDA=ON

The build process tries to find the CUDA Toolkit and the cuDNN library by
default. You can also specify the explicit locations of their libraries if
searching failed or you want to switch them:

.. code-block:: shell

  $ cmake .. \
    -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
    -DCUDNN_ROOT_DIR=/path/to/cuda \
    -DPRIMITIV_USE_CUDA=ON


Installing primitiv with OpenCL
-------------------------------


Installing OpenCL C++ Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: shell

  $ git clone https://github.com/KhronosGroup/OpenCL-CLHPP.git
  $ cd OpenCL-CLHPP
  $ mkdir build
  $ cd build
  $ cmake .. [OPTIONS]  # See: https://github.com/KhronosGroup/OpenCL-CLHPP
  $ make && make install


Installing CLBlast
^^^^^^^^^^^^^^^^^^


.. code-block:: shell

  $ apt install wget
  $ wget https://github.com/CNugteren/CLBlast/archive/1.2.0.tar.gz -O ./clblast.tar.gz
  $ mkdir clblast
  $ cd clblast
  $ tar xf ../clblast.tar.gz --strip-components 1
  $ mkdir build
  $ cd build
  $ cmake .. [OPTIONS]  # See: https://github.com/CNugteren/CLBlast
  $ make && make install


Configuring primitiv with OpenCL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The following command configures to build the OpenCL backend using
system libraries.

.. code-block:: shell

  $ cmake .. -DPRIMITIV_USE_OPENCL=ON

The build process tries to find the OpenCL library, the OpenCL C++ headers,
and the CLBlast library by default. You can also specify the explicit
locations of their libraries if searching failed or you want to switch them:

.. code-block:: shell

  $ cmake .. \
    -DOpenCL_INCLUDE_DIR=/path/to/opencl/include \
    -DOpenCL_LIBRARY=/path/to/libOpenCL.so \
    -DCLHPP_INCLUDE_DIR=/path/to/clhpp/include \
    -DCLBLAST_ROOT=/path/to/clblast/prefix \
    -DPRIMITIV_USE_OPENCL=ON
