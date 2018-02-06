======================================
Headers, Library Files and Compilation
======================================


Compile Options
---------------


primitiv is written in **C++11**. Users must specify appropriate compiler
options to enable the C++11 specification.
In most GCC-like compilers, ``-std=c++11`` option can be used for this purpose.


Install Paths
-------------


primitiv is installed according to the usual process of CMake.
In most UNIX-like systems, all files required to use primitiv is installed into
``/usr/local`` by default. Users can changethe this location using
``CMAKE_INSTALL_PREFIX`` standard option of CMake.

After installation, the install location should have at least following files::

  PREFIX/include/primitiv/primitiv.h
                         / ... (other files)
                         /c/api.h
                           / ... (other files)
        /lib/libprimitiv.so


Header Files
------------


All C++ header files of primitiv is placed in the ``PREFIX/include/primitiv``
directory.
``primitiv.h`` is a useful header to include all features of primitiv installed
onto your machine. Whether some features (e.g. CUDA device class) can be used or
not is represented as the macros defined in ``config.h``

``PREFIX/include/primitiv/c`` directory stores C-language API headers used by
some bindings between other languages. ``c/api.h`` can be used similarly with
``primitiv.h`` to include all the features available through C APIs.

If the ``PREFIX`` directory is specified as an root of the include paths, you
can include these header files like following:

.. code-block:: c++

  #include <primitiv/primitiv.h>
  #include <primitiv/c/api.h>


Library Files
-------------


``PREFIX/lib`` directory has ``libprimitiv.so`` shared object file.
Users should link this file when compiling your own code using primitiv.

If the ``PREFIX`` directory is specified as an root of the library paths, you
can link ``libprimitiv.so`` like following:

.. code-block:: shell

  cc -std=c++11 your_source.cc -lprimitiv
