#include <config.h>

#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/c/functions.h>
#include <primitiv/c/graph.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/c/naive_device.h>
#include <primitiv/operator_impl.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv_c {

class CGraphTest : public testing::Test {
  void SetUp() override {
    dev = ::primitiv_Naive_new();
    dev2 = ::primitiv_Naive_new();
  }
  void TearDown() override {
    ::primitiv_Naive_delete(dev);
    ::primitiv_Naive_delete(dev2);
  }
 protected:
  ::primitiv_Device *dev;
  ::primitiv_Device *dev2;
};

TEST_F(CGraphTest, CheckDefault) {
  ::primitiv_Graph *graph;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Graph_get_default(&graph));
  {
    ::primitiv_Graph *g1 = ::primitiv_Graph_new();
    ::primitiv_Graph_set_default(g1);
    ::primitiv_Graph_get_default(&graph);
    EXPECT_EQ(g1, graph);
    {
      ::primitiv_Graph *g2 = ::primitiv_Graph_new();
      ::primitiv_Graph_set_default(g2);
      ::primitiv_Graph_get_default(&graph);
      EXPECT_EQ(g2, graph);
      ::primitiv_Graph_delete(g2);
    }
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Graph_get_default(&graph));
    ::primitiv_Graph *g3 = ::primitiv_Graph_new();
    ::primitiv_Graph_set_default(g3);
    ::primitiv_Graph_get_default(&graph);
    EXPECT_EQ(g3, graph);
    ::primitiv_Graph_delete(g1);
    ::primitiv_Graph_delete(g3);
  }
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Graph_get_default(&graph));
}

TEST_F(CGraphTest, CheckInvalidNode) {
  ::primitiv_Node *node = ::primitiv_Node_new();
  EXPECT_FALSE(::primitiv_Node_valid(node));
  ::primitiv_Graph *graph;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_graph(node, &graph));
  uint32_t id;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_operator_id(node, &id));
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_value_id(node, &id));
  ::primitiv_Shape *shape;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_shape(node, &shape));
  ::primitiv_Device *device;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_device(node, &device));
  float value;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_to_float(node, &value));
  float values[20];
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_to_array(node, values));
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Node_backward(node));
  ::primitiv_Node_delete(node);
}

TEST_F(CGraphTest, CheckClear) {
  ::primitiv_Device_set_default(dev);

  ::primitiv_Graph *g = ::primitiv_Graph_new();
  ::primitiv_Graph_set_default(g);

  EXPECT_EQ(0u, ::primitiv_Graph_num_operators(g));

  {
    ::primitiv_Shape *shape = ::primitiv_Shape_new();
    float values[] = {1};
    ::primitiv_Node *node1;
    ::primitiv_node_func_input(shape, values, 1, nullptr, nullptr, &node1);
    ::primitiv_Node_delete(node1);
    ::primitiv_Node *node2;
    ::primitiv_node_func_input(shape, values, 1, nullptr, nullptr, &node2);
    ::primitiv_Node_delete(node2);
    EXPECT_EQ(2u, ::primitiv_Graph_num_operators(g));
  }

  ::primitiv_Graph_clear(g);
  EXPECT_EQ(0u, ::primitiv_Graph_num_operators(g));

  {
    ::primitiv_Shape *shape = ::primitiv_Shape_new();
    float values[] = {1};
    ::primitiv_Node *node;
    ::primitiv_node_func_input(shape, values, 1, nullptr, nullptr, &node);
    ::primitiv_Node_delete(node);
    ::primitiv_node_func_input(shape, values, 1, nullptr, nullptr, &node);
    ::primitiv_Node_delete(node);
    ::primitiv_node_func_input(shape, values, 1, nullptr, nullptr, &node);
    ::primitiv_Node_delete(node);
    EXPECT_EQ(3u, ::primitiv_Graph_num_operators(g));
  }

  ::primitiv_Graph_clear(g);
  EXPECT_EQ(0u, ::primitiv_Graph_num_operators(g));

  // Clear empty graph.
  ::primitiv_Graph_clear(g);
  EXPECT_EQ(0u, ::primitiv_Graph_num_operators(g));

  ::primitiv_Graph_delete(g);
}

/*
TEST_F(GraphTest, CheckForward) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data3 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  vector<Node> nodes;
  nodes.emplace_back(functions::input<Node>(Shape({2, 2}, 3), data1));
  nodes.emplace_back(functions::ones<Node>({2, 2}));
  nodes.emplace_back(functions::input<Node>(Shape({2, 2}, 3), data3));
  nodes.emplace_back(nodes[0] + nodes[1]);
  nodes.emplace_back(nodes[1] - nodes[2]);
  nodes.emplace_back(nodes[3] * nodes[4]);
  nodes.emplace_back(nodes[5] + 1);
  nodes.emplace_back(functions::sum(nodes[6], 0));
  nodes.emplace_back(functions::sum(nodes[7], 1));
  nodes.emplace_back(functions::batch::sum(nodes[8]));

  EXPECT_EQ(10u, nodes.size());
  EXPECT_EQ(10u, g.num_operators());

  // Dump the graph to the output log.
  std::cout << g.dump("dot");

  // Check all shapes and devices.
  const vector<Shape> expected_shapes {
    Shape({2, 2}, 3), {2, 2}, Shape({2, 2}, 3),
    Shape({2, 2}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3),
    Shape({2, 2}, 3),
    Shape({1, 2}, 3), Shape({}, 3), {},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(expected_shapes[i], nodes[i].shape());
    EXPECT_EQ(&dev, &nodes[i].device());
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
    {7, 11, 2, 2, -3, -7},
    {18, 4, -10},
    {12},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    // This forward method has no effect and only returns the reference to the
    // inner value.
    const Tensor &val = g.forward(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val.to_vector()));
    EXPECT_TRUE(vector_match(expected_values[i], nodes[i].to_vector()));
  }
}
 */

}  // namespace primitiv_c
