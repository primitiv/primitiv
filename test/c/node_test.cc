#include <primitiv/config.h>

#include <vector>

#include <gtest/gtest.h>

#include <primitiv/c/devices/naive/device.h>
#include <primitiv/c/functions.h>
#include <primitiv/c/graph.h>
#include <primitiv/c/status.h>

#include <test_utils.h>

namespace primitiv {
namespace c {

class CNodeTest : public testing::Test {
  void SetUp() override {
    ::primitivCreateNaiveDevice(&dev);
    ::primitivCreateGraph(&g);
    ::primitivSetDefaultDevice(dev);
    ::primitivSetDefaultGraph(g);
  }
  void TearDown() override {
    ::primitivDeleteDevice(dev);
    ::primitivDeleteGraph(g);
  }
 protected:
  ::primitivDevice_t *dev;
  ::primitivGraph_t *g;
};

TEST_F(CNodeTest, CheckCopy) {
  const float data[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const uint32_t dims[] = {2, 2};
  ::primitivShape_t *shape;
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivCreateShapeWithDims(dims, 2, 3, &shape));
  ::primitivNode_t *a;
  ::primitivApplyNodeInput(shape, data, 12, nullptr, nullptr, &a);
  ASSERT_EQ(PRIMITIV_C_OK,
            ::primitivApplyNodeInput(shape, data, 12, nullptr, nullptr, &a));
  PRIMITIV_C_BOOL valid;
  ::primitivIsValidNode(a, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);

  ::primitivNode_t *b;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCloneNode(a, &b));
  ::primitivIsValidNode(a, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);
  ::primitivIsValidNode(b, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);
  ::primitivShape_t *s1;
  ::primitivShape_t *s2;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivGetNodeShape(a, &s1));
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivGetNodeShape(b, &s2));
  ::primitivIsShapeEqualTo(s1, s2, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);

  ::primitivDeleteShape(shape);
  ::primitivDeleteShape(s1);
  ::primitivDeleteShape(s2);
  ::primitivDeleteNode(a);
  ::primitivDeleteNode(b);
}

}  // namespace c
}  // namespace primitiv
