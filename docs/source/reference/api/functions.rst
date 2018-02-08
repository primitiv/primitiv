=========
Functions
=========


This page describes the basic/composite functions implemented in primitiv.
They return a template type ``Var``, and take 0 or more number of references
of ``Var`` as their arguments. ``Var`` becomes either ``Node`` or ``Tensor``
according to the usage:

.. code-block:: c++

  primitiv::Node x = ...;
  primitiv::Tensor w = ...;
  auto y = primitiv::functions::tanh(x);  // `y` becomes a `Node`.
  auto u = primitiv::functions::exp(w);  // `u` becomes a `Tensor`.

If the function has no argument with type ``Var``, you must specify the template
argument appropriately:

.. code-block:: c++

  auto x = primitiv::functions::input<Node>(...);  // `x` becomes a `Node`.
  auto w = primitiv::functions::parameter<Tensor>(...);  // `w` becomes a `Tensor`.

.. doxygennamespace:: primitiv::functions
  :members:
