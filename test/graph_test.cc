#include <config.h>

#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/node_ops.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class GraphTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(GraphTest, CheckForwardBackward) {
  Graph g;
  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data2 {1, 1, 1, 1};
  const vector<float> data3 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  vector<Node> nodes;
  nodes.emplace_back(node_ops::input(g, dev, Shape({2, 2}, 3), data1));
  nodes.emplace_back(node_ops::input(g, dev, {2, 2}, data2));
  nodes.emplace_back(node_ops::input(g, dev, Shape({2, 2}, 3), data3));
  nodes.emplace_back(nodes[0] + nodes[1]);
  nodes.emplace_back(nodes[1] - nodes[2]);
  nodes.emplace_back(nodes[3] * nodes[4]);
  nodes.emplace_back(nodes[5] + 1);

  EXPECT_EQ(nodes.size(), g.num_nodes());

  // Dump the graph to the output log.
  g.dump();

  // Check all node values are still invalid.
  for (const Node &node : nodes) {
    EXPECT_FALSE(g.get_value(node).valid());
  }

  g.forward(nodes.back());

  // Check all node values.
  const vector<vector<float>> expected_values {
    {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
    {1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
    {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5},
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1},
    {2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5},
    {3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4},
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    // This forward method has no effect and only returns the reference to the
    // inner value.
    const Tensor &val1 = g.forward(nodes[i]);
    const Tensor &val2 = g.get_value(nodes[i]);
    EXPECT_EQ(&val1, &val2);
    ASSERT_TRUE(val1.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val1.get_values()));
  }

  // Check all node gradients are still invalid.
  for (const Node &node : nodes) {
    EXPECT_FALSE(g.get_gradient(node).valid());
  }

  g.backward(nodes.back());

  // Check all node gradients.
  const vector<vector<float>> expected_grads {
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1}, // n[1] - n[2]
    {6, 9, 12, 15}, // batch_sum(n[0] + 2*n[1] - n[2])
    {-2, -3, -4, -5, -2, -3, -4, -5, -2, -3, -4, -5}, // -n[0] - n[1]
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1}, // n[4]
    {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5}, // n[3]
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 1
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 1
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.get_gradient(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_grads[i], val.get_values()));
  }
}

TEST_F(GraphTest, TestXor) {
  // Solves a 2-dimension XOR problem with 3-layer perceptron.
  // h = tanh(W1.x + b1)
  // y = W2.h + b2
  Parameter w1({2, 2}, &dev);
  Parameter b1({2}, &dev);
  Parameter w2({1, 2}, &dev);
  Parameter b2({}, &dev);

  w1.reset_value({1, -1, 1, -1});
  b1.reset_value({-1, -1});
  w2.reset_value({1, 1});
  b2.reset_value({1});

  const vector<float> inputs {1, 1, 1, -1, -1, 1, -1, -1};
  const vector<float> outputs {1, -1, -1, 1};

  Graph g;
  vector<Node> nodes;
  // sources
  nodes.emplace_back(node_ops::input(g, dev, Shape({2}, 4), inputs));
  nodes.emplace_back(node_ops::parameter(g, w1));
  nodes.emplace_back(node_ops::parameter(g, b1));
  nodes.emplace_back(node_ops::parameter(g, w2));
  nodes.emplace_back(node_ops::parameter(g, b2));
  // calculation
  nodes.emplace_back(node_ops::dot(nodes[1], nodes[0]));
  nodes.emplace_back(nodes[5] + nodes[2]);
  nodes.emplace_back(node_ops::tanh(nodes[6]));
  nodes.emplace_back(node_ops::dot(nodes[3], nodes[7]));
  nodes.emplace_back(nodes[8] + nodes[4]);
  // losses
  nodes.emplace_back(node_ops::input(g, dev, Shape({}, 4), outputs));
  nodes.emplace_back(nodes[9] - nodes[10]);
  nodes.emplace_back(nodes[11] * nodes[11]);

  EXPECT_EQ(nodes.size(), g.num_nodes());
  g.dump();

  g.forward(nodes.back());

  // Check all node values.
  const float h1 = .76159416;  // tanh(1)
  const float h2 = .99505475;  // tanh(3)
  const float h3 = -.23346060;  // tanh(1) - tanh(3)
  const float h4 = -1.5231883;  // -2 * tanh(1)
  const float h5 = .76653940;  // 1 + tanh(1) - tanh(3)
  const float h6 = -.52318831;  // 1 - 2 * tanh(1)
  const float h7 = .47681169;  // 2 - 2 * tanh(1)
  const vector<vector<float>> expected_values {
    {1, 1, 1, -1, -1, 1, -1, -1},
    {1, -1, 1, -1},
    {-1, -1},
    {1, 1},
    {1},
    {2, -2, 0, 0, 0, 0, -2, 2},
    {1, -3, -1, -1, -1, -1, -3, 1},
    {h1, -h2, -h1, -h1, -h1, -h1, -h2, h1},
    {h3, h4, h4, h3},
    {h5, h6, h6, h5},
    {1, -1, -1, 1},
    {h3, h7, h7, h3},
    {h3 * h3, h7 * h7, h7 * h7, h3 * h3},
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.get_value(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val.get_values()));
  }

  // TODO(odashi): add gradient checking.
}

}  // namespace primitiv
