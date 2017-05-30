#include <config.h>

#include <iostream>
#include <gtest/gtest.h>

#ifdef USE_CUDA
#define PRIMITIV_USE_CUDA
#endif  // USE_CUDA
#include <primitiv/primitiv.h>

namespace primitiv {

namespace F = node_ops;

TEST(ExampleTest, Xor) {
  // Solving the XOR problem with a 3-layer perceptron.

#ifdef PRIMITIV_USE_CUDA
  // Computation backend (CUDA: device ID = 0)
  CUDADevice dev(0);
#else
  // Computation backend (CPU)
  CPUDevice dev;
#endif  // USE_CUDA

  // Parameters
  Parameter pw1({8, 2}, &dev, initializers::XavierUniform());
  Parameter pb1({8}, &dev, initializers::Constant(0));
  Parameter pw2({1, 8}, &dev, initializers::XavierUniform());
  Parameter pb2({}, &dev, initializers::Constant(0));

  // Trainer
  SGDTrainer trainer(.1);
  trainer.add_parameter(&pw1);
  trainer.add_parameter(&pb1);
  trainer.add_parameter(&pw2);
  trainer.add_parameter(&pb2);

  // Input/output data
  const std::vector<float> input_data {
     1,  1,
     1, -1,
    -1,  1,
    -1, -1,
  };
  const std::vector<float> output_data {1, -1, -1, 1};

  float prev_loss = 1e10;

  // Training loop
  for (unsigned i = 0; i < 10; ++i) {
    // Builds a computation graph.
    Graph g;
    Node x = F::input(g, dev, Shape({2}, 4), input_data);
    Node w1 = F::parameter(g, pw1);
    Node b1 = F::parameter(g, pb1);
    Node w2 = F::parameter(g, pw2);
    Node b2 = F::parameter(g, pb2);
    Node h = F::tanh(F::dot(w1, x) + b1);
    Node y = F::dot(w2, h) + b2;
    Node t = F::input(g, dev, Shape({}, 4), output_data);
    Node diff = t - y;
    Node loss = F::batch_sum(diff * diff);

    // Gets results (outputs).
    std::vector<float> y_val = g.forward(y).to_vector();
    std::cout << "epoch " << i << ":" << std::endl;
    for (unsigned j = 0; j < 4; ++j) {
      std::cout << "  [" << j << "]: " << y_val[j] << std::endl;
    }

    // Gets results (losses).
    trainer.reset_gradients();
    const Tensor &loss_tensor = g.forward(loss);
    const float loss_val = loss_tensor.to_vector()[0];
    std::cout << "  loss: " << loss_val << std::endl;

    EXPECT_EQ(Shape(), loss_tensor.shape());  // Loss is a scalar.
    EXPECT_LT(loss_val, prev_loss);  // Loss always decreases.

    prev_loss = loss_val;

    // Backpropagation
    g.backward(loss);

    // Updates parameters.
    trainer.update();
  }
}

}  // namespace primitiv
