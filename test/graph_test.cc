#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/graph.h>
#include <primitiv/function_impl.h>

namespace primitiv {

class GraphTest : public testing::Test {};

TEST_F(GraphTest, CheckConstruction) {
  Graph g;
  Node n1 = g.add_function(new functions::Input(Shape({2, 2}, 1)), {});
  Node n2 = g.add_function(new functions::Input(Shape({2, 2}, 1)), {});
  Node n3 = g.add_function(new functions::Input(Shape({2, 2}, 3)), {});
  Node n4 = g.add_function(new functions::Add(), {n1, n2});
  Node n5 = g.add_function(new functions::Subtract(), {n2, n3});
  Node n6 = g.add_function(new functions::Multiply(), {n4, n5});
  g.add_function(new functions::AddConst(123), {n6});
  // for now, the numbers of value/function nodes are same because a function
  // node generates only one value node.
  EXPECT_EQ(7u, g.num_value_nodes());
  EXPECT_EQ(7u, g.num_function_nodes());
  g.dump();
}

}  // namespace primitiv
