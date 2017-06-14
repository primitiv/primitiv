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

TEST_F(ShapeOpsTest, CheckElementwise) {
  struct TestCase { Shape a, b, expected; };
  const vector<TestCase> test_cases {
    {{}, {}, {}},
    {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
    {Shape({}, 4), Shape({}, 4), Shape({}, 4)},
    {Shape({1, 2, 3}, 4), Shape({1, 2, 3}, 4), Shape({1, 2, 3}, 4)},
    {{}, Shape({}, 4), Shape({}, 4)},
    {{1, 2, 3}, Shape({1, 2, 3}, 4), Shape({1, 2, 3}, 4)},
    {Shape({}, 4), {}, Shape({}, 4)},
    {Shape({1, 2, 3}, 4), {1, 2, 3}, Shape({1, 2, 3}, 4)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_EQ(tc.expected, elementwise(tc.a, tc.b));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidElementwise) {
  struct TestCase { Shape a, b; };
  const vector<TestCase> test_cases {
    {{}, {1, 2, 3}},
    {Shape({}, 4), {1, 2, 3}},
    {{}, Shape({1, 2, 3}, 4)},
    {Shape({}, 4), Shape({1, 2, 3}, 4)},
    {Shape({}, 4), Shape({}, 5)},
    {Shape({1, 2, 3}, 4), Shape({1, 2, 3}, 5)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(elementwise(tc.a, tc.b), Error);
  }
}

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
    const Shape observed = slice(tc.input, tc.dim, tc.lower, tc.upper);
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

TEST_F(ShapeOpsTest, CheckConcat) {
  struct TestCase {
    vector<Shape> inputs;
    unsigned dim;
    Shape expected;
  };
  const vector<TestCase> test_cases {
    {{{}}, 0, {}},
    {{{}}, 1, {}},
    {{{}, {2}, {3}}, 0, {6}},
    {{{1, 2}, {3, 2}, {2, 2}}, 0, {6, 2}},
    {{{1, 2}, {1, 2, 3}, {1, 2, 2}}, 2, {1, 2, 6}},
    {{{1, 2}, {1, 2}, {1, 2}}, 3, {1, 2, 1, 3}},
    {{Shape({}, 2), {}, {}}, 0, Shape({3}, 2)},
    {{{}, Shape({2}, 2), {}}, 0, Shape({4}, 2)},
    {{{}, {}, Shape({3}, 2)}, 0, Shape({5}, 2)},
    {{Shape({}, 3), Shape({2}, 3), {3}}, 0, Shape({6}, 3)},
  };
  for (const TestCase &tc : test_cases) {
    vector<const Shape *>xs;
    for (const Shape &x : tc.inputs) xs.emplace_back(&x);
    const Shape observed = concat(xs, tc.dim);
    EXPECT_EQ(tc.expected, observed);
  }
}

TEST_F(ShapeOpsTest, CheckInvalidConcat) {
  struct TestCase {
    vector<Shape> inputs;
    unsigned dim;
  };
  const vector<TestCase> test_cases {
    {{}, 0},
    {{{1}, {2}}, 1},
    {{{1, 2}, {1, 3}}, 0},
    {{{1, 2}, {1, 3}}, 2},
    {{Shape({}, 2), Shape({}, 3)}, 0},
    {{Shape({1, 2}, 2), Shape({1, 3}, 3)}, 0},
  };
  for (const TestCase &tc : test_cases) {
    vector<const Shape *>xs;
    for (const Shape &x : tc.inputs) xs.emplace_back(&x);
    EXPECT_THROW(concat(xs, tc.dim), Error);
  }
}

}  // namespace shape_ops
}  // namespace primitiv
