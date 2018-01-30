=============================================
Step-by-step Example: Solving the XOR Problem
=============================================


This tutorial introduces a basic and common usage of the primitiv by making and
training a simple network for a small classification problem.


Introduction: Problem Formulation
---------------------------------


Following lines are the formulation of the problem used in this tutorial:

.. math::

  \begin{eqnarray}
    f: \mathbb{R}^2 & \rightarrow & [-1, 1]; \\
    f: (x_1, x_2) & \mapsto & \left\{ \begin{array}{rl}
      1, & \mathrm{if} \ \ x_1 x_2 \geq 0, \\
      -1, & \mathrm{otherwise},
    \end{array} \right.
  \end{eqnarray}

where :math:`x_1, x_2 \in \mathbb{R}`.
This is known as the *XOR problem*; :math:`f` detects whether the signs of two
arguments are same or not. This problem is *linearly non-separatable*, i.e., the
discriminator function :math:`f` can NOT be represented as an Affine transform
of given arguments:
:math:`\alpha x_1 + \beta x_2 + \gamma`,
where :math:`\alpha, \beta, \gamma \in \mathbb{R}`.

For example, following code generates random data points
:math:`(x_1 + \epsilon_1, x_2 + \epsilon_2, f(x_1, x_2))` according to this
formulation with :math:`x_1, x_2 \sim \mathcal{N}(0, \sigma_{\mathrm{data}})`
and :math:`\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \sigma_{\mathrm{noise}})`:

.. code-block:: c++

  #include <random>
  #include <tuple>

  class DataProvider {
    std::mt19937 rng;
    std::normal_distribution<float> data_dist, noise_dist;

  public:
    // Initializes the data provider with two SDs.
    DataProvider(float data_sd, float noise_sd)
      : rng(std::random_device()())
      , data_dist(0, data_sd)
      , noise_dist(0, noise_sd) {}

    // Generates a data point
    std::tuple<float, float, float> operator()() {
      const float x1 = data_dist(rng);
      const float x2 = data_dist(rng);
      return std::make_tuple(
          x1 + noise_dist(rng),    // x1 + err
          x2 + noise_dist(rng),    // x2 + err
          x1 * x2 >= 0 ? 1 : -1);  // label
    }
  };

In this tutorial, we construct a 2-layer (input-hidden-output) perceptron to
solve this problem.
The whole model formulation is:

.. math::

  y := \tanh (W_{hy} \boldsymbol{h} + b_y),

  \boldsymbol{h} := \tanh (W_{xh} \boldsymbol{x} + \boldsymbol{b}_h),

where :math:`y \in \mathbb{R}` is an output value to be fit to :math:`f(x_1, x_2)`,
:math:`\boldsymbol{x} := (x_1 \ x_2)^{\top} \in \mathbb{R}^2` is an input vector,
:math:`\boldsymbol{h} \in \mathbb{R}^N` represents the :math:`N`-dimentional
*hidden state* of the network. We use :math:`N = 8` in this tutorial.
There are also 4 free parameters: 2 matrices :math:`W_{hy} \in \mathbb{R}^{1 \times N}`
and :math:`W_{xh} \in \mathbb{R}^{N \times 2}`, and 2 bias (column) vectors
:math:`b_y \in \mathbb{R}` and :math:`\boldsymbol{b}_h \in \mathbb{R}^N`.


Include and Initialization
--------------------------


primitiv requires you to include ``primitiv/primitiv.h`` before using any
features in the source code.
All features in primitiv is enabled by including this header
(available features are depending on specified
:doc:`options while building </reference/build_options>`).

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
  
  // "Eigen" device can be enabled when -DPRIMITIV_USE_EIGEN=ON
  //devices::Eigen dev;

  // "CUDA" device can be enabled when -DPRIMITIV_USE_CUDA=ON
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
:doc:`document about default objects </design/default_object>`.

