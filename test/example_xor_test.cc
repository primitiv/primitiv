#include <config.h>

#include <iostream>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/node_ops.h>
#include <primitiv/sgd_trainer.h>

namespace primitiv {

namespace F = node_ops;

TEST(ExampleTest, Xor) {
  // Solving the XOR problem with 3-layer perceptron.
  
  // Computation backend
  CPUDevice dev;

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

  float prev_loss = 1e10;

  // Training loop.
  for (unsigned i = 0; i < 10; ++i) {
    // Build computation graph.
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

    // Get result.
    std::vector<float> y_val = g.forward(y).get_values();
    std::cout << "epoch " << i << ":" << std::endl;
    for (unsigned j = 0; j < 4; ++j) {
      std::cout << "  [" << j << "]: " << y_val[j] << std::endl;
    }

    // Backpropagation
    trainer.reset_gradients();
    float loss_val = g.forward(loss).get_values()[0];
    std::cout << "  loss: " << loss_val << std::endl;
    EXPECT_LT(loss_val, prev_loss);
    prev_loss = loss_val;
    g.backward(loss);
    trainer.update();
  }
}

}  // namespace primitiv
