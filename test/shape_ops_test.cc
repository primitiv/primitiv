#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>
#include <primitiv/shape_ops.h>

using std::vector;

namespace primitiv {
namespace shape_ops {

class ShapeOpsTest : public testing::Test {};

TEST_F(ShapeOpsTest, CheckSlice) {
  struct TestCase {
    unsigned dim, lower, upper;
    Shape input, expected;
  };
  const vector<TestCase> test_cases {
    {0, 0, 1, {}, {}}, {1, 0, 1, {}, {}},
    {0, 0, 1, {3}, {}}, {0, 1, 2, {3}, {}}, {0, 2, 3, {3}, {}},
    {0, 0, 2, {3}, {2}}, {0, 1, 3, {3}, {2}}, {0, 0, 3, {3}, {3}},
    {1, 0, 1, {3}, {3}},
    {0, 1, 2, {3, 4}, {1, 4}}, {0, 0, 3, {3, 4}, {3, 4}},
    {1, 1, 2, {3, 4}, {3}}, {1, 0, 4, {3, 4}, {3, 4}},
    {2, 0, 1, {3, 4}, {3, 4}},
  };
  for (const TestCase &tc : test_cases) {
    Shape observed = slice(tc.input, tc.dim, tc.lower, tc.upper);
    EXPECT_EQ(tc.expected, observed);
  }
}

TEST_F(ShapeOpsTest, CheckInvalidSlice) {
  struct TestCase {
    unsigned dim, lower, upper;
    Shape input;
  };
  const vector<TestCase> test_cases {
    {0, 0, 0, {}}, {0, 1, 0, {}}, {0, 0, 2, {}},
    {1, 0, 0, {}}, {1, 1, 0, {}}, {1, 1, 2, {}},
    {0, 0, 0, {2}}, {0, 0, 3, {2}}, {0, 2, 1, {2}}, {1, 0, 2, {2}},
    {0, 3, 4, {3, 4}}, {0, 3, 2, {3, 4}}, {1, 0, 5, {3, 4}}, {1, 3, 2, {3, 4}},
    {2, 0, 2, {3, 4}},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(slice(tc.input, tc.dim, tc.lower, tc.upper), Error);
  }
}

}  // namespace shape_ops
}  // namespace primitiv
