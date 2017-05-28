#include <config.h>

#include <iostream>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/node_ops.h>
#include <primitiv/sgd_trainer.h>

#ifdef USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // USE_CUDA

namespace primitiv {

namespace F = node_ops;

TEST(ExampleTest, Xor) {
  // Solving the XOR problem with a 3-layer perceptron.

#ifdef USE_CUDA
  // Computation backend (CUDA: device ID = 0)
  CUDADevice dev(0);
#else
  // Computation backend (CPU)
  CPUDevice dev;
#endif  // USE_CUDA

  // Parameters
  Parameter pw1({4, 2}, &dev, {1, 1, -1, -1, 1, -1, 1, -1});
  Parameter pb1({4}, &dev, {0, 0, 0, 0});
  Parameter pw2({1, 4}, &dev, {1, -1, -1, 1});
  Parameter pb2({}, &dev, {0});

  // Trainer
  SGDTrainer trainer(.1);
  trainer.add_parameter(&pw1);
  trainer.add_parameter(&pb1);
  trainer.add_parameter(&pw2);
  trainer.add_parameter(&pb2);

  // Training loop
  for (unsigned i = 0; i < 10; ++i) {
    // Builds a computation graph.
    Graph g;
    Node x = F::input(g, dev, Shape({2}, 4), {1, 1, 1, -1, -1, 1, -1, -1});
    Node w1 = F::parameter(g, pw1);
    Node b1 = F::parameter(g, pb1);
    Node w2 = F::parameter(g, pw2);
    Node b2 = F::parameter(g, pb2);
    Node h = F::tanh(F::dot(w1, x) + b1);
    Node y = F::dot(w2, h) + b2;
    Node t = F::input(g, dev, Shape({}, 4), {1, -1, -1, 1});
    Node diff = t - y;
    Node loss = diff * diff;

    // Gets results (outputs).
    std::vector<float> y_val = g.forward(y).to_vector();
    std::cout << "epoch " << i << ":" << std::endl;
    for (unsigned j = 0; j < 4; ++j) {
      std::cout << "  [" << j << "]: " << y_val[j] << std::endl;
    }

    // Gets results (losses).
    trainer.reset_gradients();
    float loss_val = g.forward(loss).to_vector()[0];
    std::cout << "  loss[0]: " << loss_val << std::endl;

    // Backpropagation
    g.backward(loss);

    // Updates parameters.
    trainer.update();
  }
}

}  // namespace primitiv
