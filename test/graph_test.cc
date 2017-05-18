#include <config.h>

#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/graph.h>
#include <primitiv/function_impl.h>
#include <test_utils.h>

using std::vector;

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
  nodes.emplace_back(g.add_function(
      new functions::Input(Shape({2, 2}, 3), &dev, data1), {}));
  nodes.emplace_back(g.add_function(
      new functions::Input(Shape({2, 2}, 1), &dev, data2), {}));
  nodes.emplace_back(g.add_function(
      new functions::Input(Shape({2, 2}, 3), &dev, data3), {}));
  nodes.emplace_back(g.add_function(
        new functions::Add(), {nodes[0], nodes[1]}));
  nodes.emplace_back(g.add_function(
        new functions::Subtract(), {nodes[1], nodes[2]}));
  nodes.emplace_back(g.add_function(
        new functions::Multiply(), {nodes[3], nodes[4]}));
  nodes.emplace_back(g.add_function(new functions::AddConst(1), {nodes[5]}));

  EXPECT_EQ(7u, g.num_nodes());

  // Dump the graph to the output log.
  g.dump();

  // Check all node values are still invalid.
  for (const Node &node : nodes) {
    EXPECT_FALSE(g.get_value(node).valid());
  }

  g.forward(nodes[6]);

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
    EXPECT_TRUE(val1.valid());
    EXPECT_TRUE(test_utils::vector_match(expected_values[i], val1.to_vector()));
  }

  // Check all node gradients are still invalid.
  for (const Node &node : nodes) {
    EXPECT_FALSE(g.get_gradient(node).valid());
  }

  g.backward(nodes[6]);

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
    EXPECT_TRUE(val.valid());
    EXPECT_TRUE(test_utils::vector_match(expected_grads[i], val.to_vector()));
  }
}

}  // namespace primitiv
