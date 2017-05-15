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

TEST_F(GraphTest, CheckConstruction) {
  Graph g;
  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data2 {1, 1, 1, 1};
  const vector<float> data3 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  const Node n0 = g.add_function(
      new functions::Input(Shape({2, 2}, 3), &dev, data1), {});
  const Node n1 = g.add_function(
      new functions::Input(Shape({2, 2}, 1), &dev, data2), {});
  const Node n2 = g.add_function(
      new functions::Input(Shape({2, 2}, 3), &dev, data3), {});
  const Node n3 = g.add_function(new functions::Add(), {n0, n1});
  const Node n4 = g.add_function(new functions::Subtract(), {n1, n2});
  const Node n5 = g.add_function(new functions::Multiply(), {n3, n4});
  const Node n6 = g.add_function(new functions::AddConst(1), {n5});
  // for now, the numbers of value/function nodes are same because a function
  // node generates only one value node.
  EXPECT_EQ(7u, g.num_value_nodes());
  EXPECT_EQ(7u, g.num_function_nodes());

  // Dump the graph to the output log.
  g.dump();

  // Checking each results.
  {
    // This call calculates all nodes.
    const vector<float> expected {3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4};
    const vector<float> result = g.forward(n6).to_vector();
    EXPECT_TRUE(test_utils::vector_match(expected, result));
  }
  {
    const vector<float> expected {2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5};
    const vector<float> result = g.forward(n5).to_vector();
    EXPECT_TRUE(test_utils::vector_match(expected, result));
  }
  {
    const vector<float> expected {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1};
    const vector<float> result = g.forward(n4).to_vector();
    EXPECT_TRUE(test_utils::vector_match(expected, result));
  }
  {
    const vector<float> expected {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    const vector<float> result = g.forward(n3).to_vector();
    EXPECT_TRUE(test_utils::vector_match(expected, result));
  }
}

}  // namespace primitiv
