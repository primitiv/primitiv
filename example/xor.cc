// This example shows a small usage of the primitiv library with solving the XOR
// problem by 2-layer (input/hidden/output) perceptron.
//
// Compile:
// g++
//   -std=c++11
//   -I/path/to/primitiv
//   -I/path/to/cuda/include
//   -L/path/to/primitiv/build/primitiv
//   xor.cc -lprimitiv -lprimitiv_cuda

#include <iostream>

// "primitiv.h" can be used to include most of features in primitiv.
// Users also can include individual headers to suppress compilation cost.
#include <primitiv/primitiv.h>

// "primitiv_cuda.h" is required to use CUDA backend.
#include <primitiv/primitiv_cuda.h>

// shortcuts
using std::cout;
using std::endl;
using std::vector;
using primitiv::initializers::Constant;
using primitiv::initializers::XavierUniform;
using primitiv::CUDADevice;
using primitiv::Graph;
using primitiv::Node;
using primitiv::Parameter;
using primitiv::SGDTrainer;
using primitiv::Shape;
namespace F = primitiv::node_ops;

int main() {
  // Setups a computation backend.
  // The device object manages device-specific memories, and must be destroyed
  // after all other objects were gone.
  CUDADevice dev(0);

  // Parameters
  Parameter pw1({8, 2}, &dev, XavierUniform());
  Parameter pb1({8}, &dev, Constant(0.0f));
  Parameter pw2({1, 8}, &dev, XavierUniform());
  Parameter pb2({}, &dev, Constant(0.0f));

  // Trainer
  SGDTrainer trainer(0.1f);
  trainer.add_parameter(&pw1);
  trainer.add_parameter(&pb1);
  trainer.add_parameter(&pw2);
  trainer.add_parameter(&pb2);

  // Fixed input data
  const vector<float> input_data {
     1,  1,
     1, -1,
    -1,  1,
    -1, -1,
  };

  // Corresponding output data
  const vector<float> output_data {1, -1, -1, 1};

  // Training loop
  for (unsigned i = 0; i < 10; ++i) {
    Graph g;
    
    // Builds a computation graph.
    Node x = F::input(&g, &dev, Shape({2}, 4), input_data);
    Node w1 = F::parameter(&g, &pw1);
    Node b1 = F::parameter(&g, &pb1);
    Node w2 = F::parameter(&g, &pw2);
    Node b2 = F::parameter(&g, &pb2);
    Node h = F::tanh(F::dot(w1, x) + b1);
    Node y = F::dot(w2, h) + b2;

    // Calculates values.
    vector<float> y_val = g.forward(y).to_vector();
    cout << "epoch " << i << ":" << endl;
    for (unsigned j = 0; j < 4; ++j) {
      cout << "  [" << j << "]: " << y_val[j] << endl;
    }

    // Builds an additional computation graph for the mean squared loss.
    Node t = F::input(&g, &dev, Shape({}, 4), output_data);
    Node diff = t - y;
    Node loss = F::batch_sum(diff * diff) / 4;
    
    // Calculates losses.
    // The forward() function performs over only additional paths.
    const float loss_val = g.forward(loss).to_vector()[0];
    cout << "  loss: " << loss_val << endl;

    // Resets cumulative losses of parameters.
    trainer.reset_gradients();

    // Backpropagation
    g.backward(loss);

    // Updates parameters.
    trainer.update();
  }

  return 0;
}
