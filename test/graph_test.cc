#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/graph.h>
#include <primitiv/function_impl.h>

namespace primitiv {

class GraphTest : public testing::Test {};

TEST_F(GraphTest, CheckConstruction) {
  Graph g;
  Node n1 = g.add_function(new functions::Input(Shape({2, 2}, 3)), {});
  g.add_function(new functions::Input(Shape({2, 2}, 3)), {});
  g.add_function(new functions::Input(Shape({2, 2}, 3)), {n1}); // will die
  g.dump();
  // TODO(odashi): implementation
}

}  // namespace primitiv
