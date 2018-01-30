=============================================
Step-by-step Example: Solving the XOR Problem
=============================================


This tutorial describes basic usage of the primitiv library by making and
learning a simple neural network to solve a classification problem known as the
*XOR problem*:

.. math::

  f: \mathbb{R}^2 \rightarrow [-1, 1]; \ (x, y) \mapsto \mathrm{sgn}(xy).


Include and Initialization
--------------------------


primitiv requires you to include ``primitiv/primitiv.h`` in the source code.
All features in primitiv is enabled by including this header
(available features are depending on specified
`options while building </reference/build_options>`_).

``primitiv/primitiv.h`` basically may not affect the global namespace, and all
features in the library is declared in the ``primitiv`` namespace.
But for brevity, we omit both ``std`` and ``primitiv`` namespaces in this
tutorial using the ``using namespace`` directives.
Please pay attention to this point when you reuse these snippets.

.. code-block:: c++

  #include <iostream>
  #include <vector>
  #include <primitiv/primitiv.h>

  using namespace std;
  using namespace primitiv;

  int main() {

    // All code will be described here.
    
    return 0;
  }

Before making our network, we need to create at least two objects: **Device**
and **Graph**.
**Device** objects specifies an actual computing backends (e.g., usual
CPUs, CUDA, etc.) and memory usages for these backends.
If you installed primitiv with no build options, you can initialize only
``primitiv::devices::Naive`` device object.
**Graph** objects describe a temporary computation graph constructed by your
code and provides methods to manage their graphs.

.. code-block:: c++

  devices::Naive dev;
  Graph g;
  
  // Eigen device can be enabled when -DPRIMITIV_USE_EIGEN=ON
  //devices::Eigen dev;

  // CUDA device can be enabled when -DPRIMITIV_USE_CUDA=ON
  //devices::CUDA dev(gpu_id);


Note that Device and Graph is not a singleton; you can also create any number of
Device/Graph objects if necessary (even multiple devices share the same
backend).

After initializing a Device and a Graph, we set them as the **default
device/graph** used in the library.

.. code-block:: c++

  Device::set_default(dev);
  Graph::set_default(g);

For now, it is enough to know that these are just techniques to reduce coding
efforts, and we don't touch the details of ths function.
For more details, please read the
`tutorial about default objects </tutorial/design/default_object>`_.

