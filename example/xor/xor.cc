// This example shows a small usage of the primitiv library with solving the XOR
// problem by 2-layer (input/hidden/output) perceptron.
//
// Compile:
// g++
//   -std=c++11
//   -I/path/to/primitiv/includes (typically -I../..)
//   -L/path/to/primitiv/libs     (typically -L../../build/primitiv)
//   xor.cc -lprimitiv

#include <iostream>

// "primitiv.h" can be used to include most of features in primitiv.
// Users also can include individual headers to suppress compilation cost.
#include <primitiv/primitiv.h>

// "primitiv_cuda.h" is required to use CUDA backend.
//#include <primitiv/primitiv_cuda.h>

// shortcuts
using std::cout;
using std::endl;
using std::vector;
using primitiv::initializers::Constant;
using primitiv::initializers::XavierUniform;
using primitiv::Device;
using primitiv::CPUDevice;
//using primitiv::CUDADevice;
using primitiv::Graph;
using primitiv::Node;
using primitiv::Parameter;
using primitiv::trainers::SGD;
using primitiv::Shape;
namespace F = primitiv::node_ops;

int main() {
  // Setups a computation backend.
  // The device object manages device-specific memories, and must be destroyed
  // after all other objects were gone.
  CPUDevice dev;
  Device::set_default_device(dev);

  // If you want to use CUDA, uncomment below line (and comment out above) with
  // a specific device (GPU) ID.
  //CUDADevice dev(0);

  // Parameters
  Parameter pw1("w1", {8, 2}, XavierUniform());
  Parameter pb1("b1", {8}, Constant(0.0f));
  Parameter pw2("w2", {1, 8}, XavierUniform());
  Parameter pb2("b2", {}, Constant(0.0f));

  // Trainer
  SGD trainer(0.1f);
  trainer.add_parameter(pw1);
  trainer.add_parameter(pb1);
  trainer.add_parameter(pw2);
  trainer.add_parameter(pb2);

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
    Graph::set_default_graph(g);
    
    // Builds a computation graph.
    Node x = F::input(Shape({2}, 4), input_data);
    Node w1 = F::input(pw1);
    Node b1 = F::input(pb1);
    Node w2 = F::input(pw2);
    Node b2 = F::input(pb2);
    Node h = F::tanh(F::matmul(w1, x) + b1);
    Node y = F::matmul(w2, h) + b2;

    // Calculates values.
    vector<float> y_val = g.forward(y).to_vector();
    cout << "epoch " << i << ":" << endl;
    for (unsigned j = 0; j < 4; ++j) {
      cout << "  [" << j << "]: " << y_val[j] << endl;
    }

    // Builds an additional computation graph for the mean squared loss.
    Node t = F::input(Shape({}, 4), output_data);
    Node diff = t - y;
    Node loss = F::batch::mean(diff * diff);
    
    // Calculates losses.
    // The forward() function performs over only additional paths.
    const float loss_val = g.forward(loss).to_vector()[0];
    cout << "  loss: " << loss_val << endl;

    // Resets cumulative gradients of all registered parameters.
    trainer.reset_gradients();

    // Backpropagation
    g.backward(loss);

    // Updates parameters.
    trainer.update();
  }

  return 0;
}
