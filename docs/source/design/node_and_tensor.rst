=================
Nodes and Tensors
=================


Nodes
-----


primitiv has two different classes to calculate newral networks: ``Node`` and
``Tensor``.
Nevertheless the basic usage of these classes are identical, the inner behavior
of them is essentially different.

A ``Node`` object behaves a reference to an intermediate result of the network.
Each ``Node`` object corresponds a ``Device`` object that represents the
phisical location of the calculated data, and a ``Graph`` object that the
intermediate result belongs to.
``Node`` objects contain only a few information to identify the corresponding
intermediate result in the ``Graph`` object, and have interfaces to communicate
the ``Graph`` object to obtain actual data.
Copying ``Node`` objects is typically a light operation.


Lazy Evaluation
^^^^^^^^^^^^^^^


Alithmetic operators between ``Node`` objects and functions defined in the
``primitiv::functions`` namespace register a new operation to the ``Graph``
object, and return a new ``Node`` object representing the result of the
new operation. Actual calculation of each operation is postponed until the
values are actually required.
Once the operation is performed, the resulting values will be cached in the
``Graph`` object to prevent duplicated calculation.


Following examples show how ``Node`` objects work:

.. code-block:: c++

  using namespace primitiv;
  namespace F = primitiv::functions;

  // Creating a `Node` object with no information: it does not point to any
  // existing data.
  const Node n0;

  // Creating a `Device` and a `Graph` and setting them as the defaults.
  devices::Naive dev;
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Creating two `Node` objects as the data sources of the computation graph.
  const Node n1 = F::input<Node>({3}, {1, 2, 3});
  const Node n2 = F::input<Node>({3}, {1, 1, 1});

  // Creating a new `Node` object representing the result of some operations.
  const Node n3 = n1 + n2;
  const Node n33 = F::tanh(n1);

  // Copying a `Node` object.
  // This operation does not yield copying phisical results.
  const Node n4 = n3;

  // Obtaining the actual results corresponding to a `Node` object.
  // The `n1 + n2` operation will be actually performed here.
  // And n33 is not calculated because it is not necessary to calculate `n4`.
  const std::vector<float> values4 = n4.to_vector();  // {2, 3, 4}

  // Defining an additional operation.
  const Node n5 = n4 + F::input<Node>({3}, {3, 2, 1});

  // Obtaining the result.
  // The value represents `(n1 + n2) + {3, 2, 1}`, but the actual calculation
  // will prevent the `n1 + n2` operation, and use the cached values of `n4`.
  const std::vector<float> values5 = n5.to_vector();  // {5, 5, 5}


Executing Backpropagation
^^^^^^^^^^^^^^^^^^^^^^^^^


``Node`` object can perform the backpropagation.
Unlike the *forward* operations described above, results of the backpropagation
(*gradients* corresponding to ``Node`` objects) will be discarded whenever it is
no longer used.
To execute the backpropagation from a specified ``Node`` object (typically the
``Node`` representing the *sum of loss values*), users should call the
``Node::backward()`` function:

.. code-block:: c++

  using namespace primitiv;
  namespace F = primitiv::functions;

  devices::Naive dev;
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Creating the graph with a `Parameter`.
  Parameter p({3}, {0, 0, 0});
  const Node w = F::parameter(p);
  const Node x = F::input({3}, {1, 2, 3});
  const Node y = w * x;  // Elementwise multiplication

  // Initializes the gradients of parameters.
  p.reset_gradient();
  const std::vector grad1 = y.gradient().to_vector();  // {0, 0, 0}

  // Executing the backpropagation.
  y.backward();

  // All gradient values are disposed before arriving here.
  const std::vector grad2 = y.gradient().to_vector();  // {1, 2, 3}


Tensor
------


``Tensor`` class is another interface to calculate networks using similar
interface with ``Node``.
Unlike the ``Node`` objects, ``Tensor`` objects hold actual resulting values
of corresponding operations, and the calculation will be performed at the same
time as creating new ``Tensor`` objects.
Additionally, ``Tensor`` objects can not perform the backpropagation because
they do not record the history of calculation.

Instead of these disadvantages, ``Tensor`` objects do not consume more memory
than actual existence of all ``Tensor`` objects at the time, and do not yield
any overhead of constructing computation graphs.
Users can use ``Tensor`` instead of ``Node`` when users do not need the gradient
information (e.g., testing trained models).

Following examples show how the ``Tensor`` objects work:

.. code-block:: c++

  using namespace primitiv;
  namespace F = primitiv::functions;

  // Creating a `Tensor` object with no information: it does not point to any
  // existing data.
  const Tensor t0;

  // Creating a `Device` and setting it as the default.
  // `Tensor` objects do not require the `Graph` object.
  devices::Naive dev;
  Device::set_default(dev);

  // Creating two `Tensor` objects with their own data.
  const Tensor t1 = F::input<Tensor>({3}, {1, 2, 3});
  const Tensor t2 = F::input<Tensor>({3}, {1, 1, 1});

  // Creating a new `Tensor` object representing the result of some operations.
  // The operations will be performed as soon as these statements are evaluated.
  // And `t3` and `t33` hold their own values internally.
  const Tensor t3 = t1 + t2;
  const Tensor t33 = F::tanh(t1);

  // Copying a `Tensor` object.
  // This operation basically does not yield a large overhead.
  // `n3` and `n4` shares the inner memory while they refers the same values.
  const Tensor t4 = t3;

  // Obtaining the inner values from a `Tensor` object.
  const std::vector<float> values4 = n4.to_vector();  // {2, 3, 4}
