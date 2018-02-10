==========================
Shapes and Operation Rules
==========================


Shapes
------


Node, Tensor and Parameter objects have a **Shape** which describes actual
appearances of inner data of those objects.

Shape consists of two elements: *dimension* and *minibatch size*.
Dimensions are the list of integers which describes volumes of each axis.
For example, following code creates new Shape descriptors of a *scalar*, a
*column vector*, a *matrix* and a *image* used in CNN functions:

.. code-block:: c++

  using primitiv::Shape;
  
  // Creating Shape of scalars.
  const Shape scalar1({});
  const Shape scalar2 {};
  const Shape scalar3;

  // Creating Shape ov 3-dimentional column vectors.
  const Shape vector1({3});
  const Shape vector2 {3};

  // Creating Shape of 3x2 matrices.
  const Shape matrix1({3, 2});  // {rows, columns}
  const Shape matrix2 {3, 2};

  // Creating Shape of the image.
  const Shape image1({256, 256, 3});  // {width1, width2, channel}

  // Shapes with the minibatch size 64.
  const Shape scalar_minibatched({}, 64);
  const Shape vector_minibatched({3}, 64);
  const Shape matrix_minibatched({3, 2}, 64);
  const Shape image_minibatched({256, 256, 3}, 64);

Two Shapes can be compared using ``==`` and ``!=`` operators:

.. code-block:: c++

  using primitiv::Shape;
  using namespace std;

  const Shape shape1 {3, 2};
  const Shape shape2 {3, 2};
  const Shape shape3 {3, 3};
  const Shape shape4({3, 2}, 64);

  cout << boolalpha;
  cout << (shape1 == shape2) << endl;  // true
  cout << (shape1 == shape3) << endl;  // false
  cout << (shape1 == shape4) << endl;  // false

primitiv does not distinguish shapes by the number of dimensions.
All Shapes with smaller number of dimensions are completely *compatible* with
Shapes with arbitrary bigger number of dimensions with the size of excessive
dimensions **1**:

.. code-block:: c++

  using primitiv::Shape;
  using namespace std;

  const Shape scalar1 {};
  const Shape scalar2 {1, 1, 1, 1};

  const Shape vector1 {3};
  const Shape vector2 {3, 1};  // This looks also a 3x1 matrix.

  const Shape matrix1 {3, 2};
  const Shape matrix2 {3, 2, 1};  // This looks also a 3x2 image with 1 channel.

  cout << boolalpha;
  cout << (scalar1 == scalar2) << endl;  // true
  cout << (vector1 == vector2) << endl;  // true
  cout << (matrix1 == matrix2) << endl;  // true


Minibatch Broadcasting
----------------------


All functions that take 2 or more Nodes or Tensors applies following rules:

#. If the shapes of two variables **have the same minibatch size**,
   the function performs independently for each data in the minibatch.
#. If at least one shape of a variable **has no minibatch (= minibatch size 1)**,
   the function broadcasts values to the minibatch size of the opposite side.
#. Otherwise, the function generates an error.
#. Functions that take more than 2 Nodes or Tensors perform above rules
   recursively.

Following examples shows how these rules work.

.. code-block:: c++

  using primitiv::Node;
  namespace F = primitiv::functions;

  const Node a = F::input<Node>(Shape({}, 3), {1, 2, 3});
  
  Node b = F::input<Node>(Shape({}, 3), {4, 5, 6});
  Node y = a + b;  // values: 5, 7, 9

  b = F::input<Node>({}, {4});
  y = a + b;  // values: 5, 6, 7
  y = b + a;  // values: 5, 6, 7

  b = F::input<Node>(Shape({}, 2), {4, 5});
  y = a + b;  // Error: different minibatch sizes: 3 and 2.
  y = b + a;  // Error: different minibatch sizes: 2 and 3.

  b = F::input<Node>({}, {4});
  const Node c = F::input(Shape({}, 3), {5, 6, 7});
  y = F::concat({a, b, c}, 0);  // values: [1, 4, 5], [2, 4, 6], [3, 4, 7]


Scalar Operations
-----------------


Elementwise binary operations such as **arithmetic operations**
(``operator+``, ``operator-``, ``operator*`` and ``operator/``) and
**exponentation** (``primitiv::functions::pow``) supports the calculation
between an arbitrary and scalar shapes.
If a shape of one operand is a scalar, these functions broadcast the scalar
value to all elements in the opposite side:

.. code-block:: c++

  using primitiv::Node;
  namespace F = primitiv::functions;

  const Node a = F::input<Node>({3}, {1, 2, 3});
  const Node b = F::input<Node>({}, {4});

  Node y = a + b;  // values: [5, 6, 7]
  y = a - b;  // values: [-3, -2, -1]
  y = b - a;  // values: [3, 2, 1]
