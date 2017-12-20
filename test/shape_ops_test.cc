#include <primitiv/config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>
#include <primitiv/shape_ops.h>

using std::vector;

namespace primitiv {
namespace shape_ops {

class ShapeOpsTest : public testing::Test {};

TEST_F(ShapeOpsTest, CheckReshape) {
  struct TestCase { Shape a, b, expected; };
  const vector<TestCase> test_cases {
    {{}, {}, {}},
    {{2, 1, 2}, {2, 1, 2}, {2, 1, 2}},
    {{2, 1, 2}, {2, 2}, {2, 2}},
    {{2, 1, 2}, {4}, {4}},
    {{2, 2}, {2, 1, 2}, {2, 1, 2}},
    {{2, 2}, {2, 2}, {2, 2}},
    {{2, 2}, {4}, {4}},
    {{4}, {2, 1, 2}, {2, 1, 2}},
    {{4}, {2, 2}, {2, 2}},
    {{4}, {4}, {4}},
    {Shape({2, 1, 2}, 3), {2, 1, 2}, Shape({2, 1, 2}, 3)},
    {Shape({2, 1, 2}, 3), {2, 2}, Shape({2, 2}, 3)},
    {Shape({2, 1, 2}, 3), {4}, Shape({4}, 3)},
    {Shape({2, 2}, 3), {2, 1, 2}, Shape({2, 1, 2}, 3)},
    {Shape({2, 2}, 3), {2, 2}, Shape({2, 2}, 3)},
    {Shape({2, 2}, 3), {4}, Shape({4}, 3)},
    {Shape({4}, 3), {2, 1, 2}, Shape({2, 1, 2}, 3)},
    {Shape({4}, 3), {2, 2}, Shape({2, 2}, 3)},
    {Shape({4}, 3), {4}, Shape({4}, 3)},
    {Shape({2, 1, 2}, 3), Shape({2, 1, 2}, 3), Shape({2, 1, 2}, 3)},
    {Shape({2, 1, 2}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3)},
    {Shape({2, 1, 2}, 3), Shape({4}, 3), Shape({4}, 3)},
    {Shape({2, 2}, 3), Shape({2, 1, 2}, 3), Shape({2, 1, 2}, 3)},
    {Shape({2, 2}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3)},
    {Shape({2, 2}, 3), Shape({4}, 3), Shape({4}, 3)},
    {Shape({4}, 3), Shape({2, 1, 2}, 3), Shape({2, 1, 2}, 3)},
    {Shape({4}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3)},
    {Shape({4}, 3), Shape({4}, 3), Shape({4}, 3)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_EQ(tc.expected, reshape(tc.a, tc.b));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidReshape) {
  struct TestCase { Shape a, b; };
  const vector<TestCase> test_cases {
    {{2}, {3}},
    {{}, Shape({}, 4)},
    {{2}, Shape({3}, 4)},
    {Shape({2}, 4), Shape({3}, 4)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(reshape(tc.a, tc.b), Error);
  }
}

TEST_F(ShapeOpsTest, CheckFlatten) {
  struct TestCase { Shape x, expected; };
  const vector<TestCase> test_cases {
    {{}, {}},
    {{2}, {2}},
    {{2, 2}, {4}},
    {{2, 1, 2}, {4}},
    {Shape({}, 3), Shape({}, 3)},
    {Shape({2}, 3), Shape({2}, 3)},
    {Shape({2, 2}, 3), Shape({4}, 3)},
    {Shape({2, 1, 2}, 3), Shape({4}, 3)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_EQ(tc.expected, flatten(tc.x));
  }
}

TEST_F(ShapeOpsTest, CheckScalarOp) {
  struct TestCase { Shape a, b, expected; };
  const vector<TestCase> test_cases {
    {{}, {}, {}},
    {{2}, {}, {2}},
    {{}, Shape({}, 4), Shape({}, 4)},
    {{2}, Shape({}, 4), Shape({2}, 4)},
    {Shape({}, 3), {}, Shape({}, 3)},
    {Shape({2}, 3), {}, Shape({2}, 3)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_EQ(tc.expected, scalar_op(tc.a, tc.b));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidScalarOp) {
  struct TestCase { Shape a, b; };
  const vector<TestCase> test_cases {
    {{}, {2}},
    {{2}, {2}},
    {{}, Shape({2}, 4)},
    {{2}, Shape({2}, 4)},
    {Shape({}, 3), {2}},
    {Shape({2}, 3), {2}},
    {Shape({}, 3), Shape({}, 4)},
    {Shape({2}, 3), Shape({}, 4)},
    {Shape({}, 3), Shape({2}, 4)},
    {Shape({2}, 3), Shape({2}, 4)},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(scalar_op(tc.a, tc.b), Error);
  }
}

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
    std::uint32_t dim, lower, upper;
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
    std::uint32_t dim, lower, upper;
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
    std::uint32_t dim;
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
    {
      // Using objects
      vector<Shape> xs;
      for (const Shape &x : tc.inputs) xs.emplace_back(x);
      const Shape observed = concat(xs, tc.dim);
      EXPECT_EQ(tc.expected, observed);
    }
    {
      // Using pointers
      vector<const Shape *> xs;
      for (const Shape &x : tc.inputs) xs.emplace_back(&x);
      const Shape observed = concat(xs, tc.dim);
      EXPECT_EQ(tc.expected, observed);
    }
  }
}

TEST_F(ShapeOpsTest, CheckInvalidConcat) {
  struct TestCase {
    vector<Shape> inputs;
    std::uint32_t dim;
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

TEST_F(ShapeOpsTest, CheckPick) {
  struct TestCase {
    Shape input;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    Shape expected;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2, 2}, 3), 0, {0, 0, 0}, Shape({1, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), 0, {1, 0, 1}, Shape({1, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), 0, {0}, Shape({1, 2, 2}, 3)},
    {{2, 2, 2}, 0, {0, 1, 0}, Shape({1, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), 1, {0, 0, 0}, Shape({2, 1, 2}, 3)},
    {Shape({2, 2, 2}, 3), 2, {0, 0, 0}, Shape({2, 2, 1}, 3)},
  };
  for (const TestCase &tc : test_cases) {
    const Shape observed = pick(tc.input, tc.ids, tc.dim);
    EXPECT_EQ(tc.expected, observed);
  }
}

TEST_F(ShapeOpsTest, CheckInvalidPick) {
  struct TestCase {
    Shape input;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
  };
  const vector<TestCase> test_cases {
     {Shape({2, 2, 2}, 3), 0, {}},
     {Shape({2, 2, 2}, 3), 0, {2}},
     {Shape({2, 2, 2}, 3), 0, {0, 1}},
     {Shape({2, 2, 2}, 3), 0, {0, 1, 2}},
     {Shape({2, 2, 2}, 3), 1, {2}},
     {Shape({2, 2, 2}, 3), 2, {2}},
     {Shape({2, 2, 2}, 3), 3, {1}},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(pick(tc.input, tc.ids, tc.dim), Error);
  }
}

TEST_F(ShapeOpsTest, CheckTranspose) {
  EXPECT_EQ(Shape(), transpose({}));
  EXPECT_EQ(Shape({}, 5), transpose(Shape({}, 5)));
  EXPECT_EQ(Shape({2}), transpose({1, 2}));
  EXPECT_EQ(Shape({2}, 5), transpose(Shape({1, 2}, 5)));
  EXPECT_EQ(Shape({1, 2}), transpose({2}));
  EXPECT_EQ(Shape({1, 2}, 5), transpose(Shape({2}, 5)));
  EXPECT_EQ(Shape({2, 3}), transpose({3, 2}));
  EXPECT_EQ(Shape({2, 3}, 5), transpose(Shape({3, 2}, 5)));
}

TEST_F(ShapeOpsTest, CheckInvalidTranspose) {
  EXPECT_THROW(transpose({1, 1, 2}), Error);
  EXPECT_THROW(transpose(Shape({1, 1, 2}, 5)), Error);
  EXPECT_THROW(transpose({1, 2, 2}), Error);
  EXPECT_THROW(transpose(Shape({1, 2, 2}, 5)), Error);
  EXPECT_THROW(transpose({2, 3, 4}), Error);
  EXPECT_THROW(transpose(Shape({2, 3, 4}, 5)), Error);
}

TEST_F(ShapeOpsTest, CheckMatMul) {
  EXPECT_EQ(Shape(), matmul({}, {}));
  EXPECT_EQ(Shape({}, 3), matmul(Shape({}, 3), {}));
  EXPECT_EQ(Shape({}, 3), matmul({}, Shape({}, 3)));
  EXPECT_EQ(Shape({}, 3), matmul(Shape({}, 3), Shape({}, 3)));
  EXPECT_EQ(Shape({10}), matmul({10}, {}));
  EXPECT_EQ(Shape({10}, 3), matmul(Shape({10}, 3), {}));
  EXPECT_EQ(Shape({10}, 3), matmul({10}, Shape({}, 3)));
  EXPECT_EQ(Shape({10}, 3), matmul(Shape({10}, 3), Shape({}, 3)));
  EXPECT_EQ(Shape({1, 10}), matmul({}, {1, 10}));
  EXPECT_EQ(Shape({1, 10}, 3), matmul(Shape({}, 3), {1, 10}));
  EXPECT_EQ(Shape({1, 10}, 3), matmul({}, Shape({1, 10}, 3)));
  EXPECT_EQ(Shape({1, 10}, 3), matmul(Shape({}, 3), Shape({1, 10}, 3)));
  EXPECT_EQ(Shape({}), matmul({1, 10}, {10}));
  EXPECT_EQ(Shape({}, 3), matmul(Shape({1, 10}, 3), {10}));
  EXPECT_EQ(Shape({}, 3), matmul({1, 10}, Shape({10}, 3)));
  EXPECT_EQ(Shape({}, 3), matmul(Shape({1, 10}, 3), Shape({10}, 3)));
  EXPECT_EQ(Shape({10, 10}), matmul({10}, {1, 10}));
  EXPECT_EQ(Shape({10, 10}, 3), matmul(Shape({10}, 3), {1, 10}));
  EXPECT_EQ(Shape({10, 10}, 3), matmul({10}, Shape({1, 10}, 3)));
  EXPECT_EQ(Shape({10, 10}, 3), matmul(Shape({10}, 3), Shape({1, 10}, 3)));
  EXPECT_EQ(Shape({20}), matmul({20, 10}, {10}));
  EXPECT_EQ(Shape({20}, 3), matmul(Shape({20, 10}, 3), {10}));
  EXPECT_EQ(Shape({20}, 3), matmul({20, 10}, Shape({10}, 3)));
  EXPECT_EQ(Shape({20}, 3), matmul(Shape({20, 10}, 3), Shape({10}, 3)));
  EXPECT_EQ(Shape({1, 20}), matmul({1, 10}, {10, 20}));
  EXPECT_EQ(Shape({1, 20}, 3), matmul(Shape({1, 10}, 3), {10, 20}));
  EXPECT_EQ(Shape({1, 20}, 3), matmul({1, 10}, Shape({10, 20}, 3)));
  EXPECT_EQ(Shape({1, 20}, 3), matmul(Shape({1, 10}, 3), Shape({10, 20}, 3)));
  EXPECT_EQ(Shape({20, 30}), matmul({20, 10}, {10, 30}));
  EXPECT_EQ(Shape({20, 30}, 3), matmul(Shape({20, 10}, 3), {10, 30}));
  EXPECT_EQ(Shape({20, 30}, 3), matmul({20, 10}, Shape({10, 30}, 3)));
  EXPECT_EQ(Shape({20, 30}, 3), matmul(Shape({20, 10}, 3), Shape({10, 30}, 3)));
}

TEST_F(ShapeOpsTest, CheckInvalidMatMul) {
  EXPECT_THROW(matmul({1, 1, 2}, {2}), Error);
  EXPECT_THROW(matmul({}, {1, 1, 2}), Error);
  EXPECT_THROW(matmul({2, 3}, {4, 5}), Error);
  EXPECT_THROW(matmul(Shape({}, 2), Shape({}, 3)), Error);
}

}  // namespace shape_ops
}  // namespace primitiv
