#include <config.h>

#include <gtest/gtest.h>

#include <primitiv/cpu_device.h>
#include <primitiv/node_ops.h>

namespace primitiv {

namespace F = node_ops;

TEST(ExampleTest, Xor) {
  // Solving the XOR problem with 3-layer perceptron.
  
  // Computation backend
  CPUDevice dev;

  // Parameters
  Parameter pw1({2, 2}, &dev);
  Parameter pb1({2}, &dev);
  Parameter pw2({1, 2}, &dev);
  Parameter pb2({}, &dev);
  pw1.reset_value({1, -1, 1, -1});
  pb1.reset_value({-1, -1});
  pw2.reset_value({1, 1});
  pb2.reset_value({1});

  // Build computation graph.
  Graph g;
  Node x = F::input(g, dev, Shape({2}, 4), {1, 1, 1, -1, -1, 1, -1, -1});
  Node w1 = F::parameter(g, pw1);
  Node b1 = F::parameter(g, pb1);
  Node w2 = F::parameter(g, pw2);
  Node b2 = F::parameter(g, pb2);
  
  Node h = F::tanh(F::dot(w1, x) + b1);
  Node y = F::dot(w2, h) + b2;

  // Get result.
  std::vector<float> y_val = g.forward(y).get_values();
  EXPECT_FLOAT_EQ(.76653940, y_val[0]);
  EXPECT_FLOAT_EQ(-.52318831, y_val[1]);
  EXPECT_FLOAT_EQ(-.52318831, y_val[2]);
  EXPECT_FLOAT_EQ(.76653940, y_val[3]);
}

}  // namespace primitiv
