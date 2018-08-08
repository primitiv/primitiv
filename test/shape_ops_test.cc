#include <primitiv/config.h>

#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/shape_ops.h>

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
      const Shape observed = concat(tc.inputs, tc.dim);
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
    {{{}, {2}}, 1},
    {{{1, 2}, {1, 3}}, 0},
    {{{1, 2}, {1, 3}}, 2},
    {{Shape({}, 2), Shape({}, 3)}, 0},
    {{Shape({1, 2}, 2), Shape({1, 3}, 3)}, 0},
  };
  for (const TestCase &tc : test_cases) {
    // Using objects
    EXPECT_THROW(concat(tc.inputs, tc.dim), Error);
    // Using pointers
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

TEST_F(ShapeOpsTest, CheckPermuteDims) {
  EXPECT_EQ(Shape({}, 2), permute_dims(Shape({}, 2), {}));
  EXPECT_EQ(Shape({}, 2), permute_dims(Shape({}, 2), {0}));
  EXPECT_EQ(Shape({2}, 3), permute_dims(Shape({2}, 3), {0}));
  EXPECT_EQ(Shape({2, 3}, 2), permute_dims(Shape({2, 3}, 2), {0, 1}));
  EXPECT_EQ(Shape({3, 2}, 3), permute_dims(Shape({2, 3}, 3), {1, 0}));
  EXPECT_EQ(Shape({2, 3, 5}, 2), permute_dims(Shape({2, 3, 5}, 2), {0, 1, 2}));
  EXPECT_EQ(Shape({5, 1, 3}, 3), permute_dims(Shape({3, 5}, 3), {1, 2, 0}));
  EXPECT_EQ(Shape({5, 2, 3}, 2), permute_dims(Shape({2, 3, 5}, 2), {2, 0, 1}));
  EXPECT_EQ(Shape({1, 3, 2}, 3), permute_dims(Shape({2, 3}, 3), {2, 1, 0}));
  EXPECT_EQ(Shape({3, 2, 5}, 2), permute_dims(Shape({2, 3, 5}, 2), {1, 0, 2}));
  EXPECT_EQ(Shape({1, 1, 1, 1, 3, 2, 7, 5}, 3), permute_dims(Shape({2, 3, 5, 7}, 3), {5, 4, 7, 6, 1, 0, 3, 2}));
  EXPECT_EQ(Shape({2, 3, 5, 7, 3, 4, 8, 10}, 2), permute_dims(Shape({2, 3, 5, 7, 3, 4, 8, 10}, 2), {0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(Shape({4, 3, 10, 8, 3, 2, 7, 5}, 3), permute_dims(Shape({2, 3, 5, 7, 3, 4, 8, 10}, 3), {5, 4, 7, 6, 1, 0, 3, 2}));
}

TEST_F(ShapeOpsTest, CheckInvalidPermuteDims) {
  EXPECT_THROW(permute_dims(Shape({2}, 3), {}), Error);
  EXPECT_THROW(permute_dims(Shape({2, 3}, 2), {0, 2}), Error);
  EXPECT_THROW(permute_dims(Shape({2, 3}, 3), {0}), Error);
  EXPECT_THROW(permute_dims(Shape({2, 3, 5}, 2), {4, 0, 1, 2}), Error);
  EXPECT_THROW(permute_dims(Shape({2, 3, 5, 7, 3, 4, 8, 10}, 2), {0, 1, 2, 3, 4, 5, 6, 7, 8}), Error);
}

TEST_F(ShapeOpsTest, CheckMatMul) {
  struct TestCase {
    vector<std::uint32_t> a, b, y;
  };
  const vector<TestCase> test_cases {
    {{}, {}, {}},
    {{10}, {}, {10}},
    {{}, {1, 10}, {1, 10}},
    {{1, 10}, {10}, {}},
    {{10}, {1, 10}, {10, 10}},
    {{20, 10}, {10}, {20}},
    {{1, 10}, {10, 20}, {1, 20}},
    {{20, 10}, {10, 30}, {20, 30}},
  };

  for (const auto &tc : test_cases) {
    EXPECT_EQ(Shape(tc.y), matmul(tc.a, tc.b));
    EXPECT_EQ(Shape(tc.y, 3), matmul(Shape(tc.a, 3), tc.b));
    EXPECT_EQ(Shape(tc.y, 3), matmul(tc.a, Shape(tc.b, 3)));
    EXPECT_EQ(Shape(tc.y, 3), matmul(Shape(tc.a, 3), Shape(tc.b, 3)));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidMatMul) {
  EXPECT_THROW(matmul({1, 1, 2}, {2}), Error);
  EXPECT_THROW(matmul({}, {1, 1, 2}), Error);
  EXPECT_THROW(matmul({2, 3}, {4, 5}), Error);
  EXPECT_THROW(matmul(Shape({}, 2), Shape({}, 3)), Error);
}

TEST_F(ShapeOpsTest, CheckConv2D) {
  struct TestCase {
    vector<std::uint32_t> x, w;
    std::uint32_t pad0, pad1, str0, str1, dil0, dil1;
    vector<std::uint32_t> y;
  };
  const vector<TestCase> test_cases {
    {{}, {}, 0, 0, 1, 1, 1, 1, {}},
    {{7, 8}, {}, 0, 0, 1, 1, 1, 1, {7, 8}},
    {{7, 8}, {3, 2}, 0, 0, 1, 1, 1, 1, {5, 7}},
    {{7, 8}, {7, 8}, 0, 0, 1, 1, 1, 1, {}},
    {{7, 8}, {1, 1, 1, 20}, 0, 0, 1, 1, 1, 1, {7, 8, 20}},
    {{7, 8}, {3, 2, 1, 20}, 0, 0, 1, 1, 1, 1, {5, 7, 20}},
    {{7, 8}, {7, 8, 1, 20}, 0, 0, 1, 1, 1, 1, {1, 1, 20}},
    {{7, 8, 10}, {1, 1, 10}, 0, 0, 1, 1, 1, 1, {7, 8}},
    {{7, 8, 10}, {3, 2, 10}, 0, 0, 1, 1, 1, 1, {5, 7}},
    {{7, 8, 10}, {7, 8, 10}, 0, 0, 1, 1, 1, 1, {}},
    {{7, 8, 10}, {1, 1, 10, 20}, 0, 0, 1, 1, 1, 1, {7, 8, 20}},
    {{7, 8, 10}, {3, 2, 10, 20}, 0, 0, 1, 1, 1, 1, {5, 7, 20}},
    {{7, 8, 10}, {7, 8, 10, 20}, 0, 0, 1, 1, 1, 1, {1, 1, 20}},
    // with padding
    {{7, 8}, {1, 1}, 1, 0, 1, 1, 1, 1, {9, 8}},
    {{7, 8}, {1, 1}, 0, 1, 1, 1, 1, 1, {7, 10}},
    {{7, 8}, {1, 1}, 1, 1, 1, 1, 1, 1, {9, 10}},
    {{7, 8}, {3, 2}, 1, 0, 1, 1, 1, 1, {7, 7}},
    {{7, 8}, {3, 2}, 0, 1, 1, 1, 1, 1, {5, 9}},
    {{7, 8}, {3, 2}, 1, 1, 1, 1, 1, 1, {7, 9}},
    {{7, 8}, {9, 8}, 1, 0, 1, 1, 1, 1, {}},
    {{7, 8}, {7, 10}, 0, 1, 1, 1, 1, 1, {}},
    {{7, 8}, {9, 10}, 1, 1, 1, 1, 1, 1, {}},
    // with stride
    {{7, 8}, {1, 1}, 0, 0, 2, 1, 1, 1, {4, 8}},
    {{7, 8}, {1, 1}, 0, 0, 1, 2, 1, 1, {7, 4}},
    {{7, 8}, {1, 1}, 0, 0, 2, 2, 1, 1, {4, 4}},
    {{7, 8}, {3, 2}, 0, 0, 2, 1, 1, 1, {3, 7}},
    {{7, 8}, {3, 2}, 0, 0, 1, 2, 1, 1, {5, 4}},
    {{7, 8}, {3, 2}, 0, 0, 2, 2, 1, 1, {3, 4}},
    {{7, 8}, {7, 8}, 0, 0, 2, 1, 1, 1, {}},
    {{7, 8}, {7, 8}, 0, 0, 1, 2, 1, 1, {}},
    {{7, 8}, {7, 8}, 0, 0, 2, 2, 1, 1, {}},
    // with dilation
    {{7, 8}, {1, 1}, 0, 0, 1, 1, 2, 1, {7, 8}},
    {{7, 8}, {1, 1}, 0, 0, 1, 1, 1, 2, {7, 8}},
    {{7, 8}, {1, 1}, 0, 0, 1, 1, 2, 2, {7, 8}},
    {{7, 8}, {3, 2}, 0, 0, 1, 1, 2, 1, {3, 7}},
    {{7, 8}, {3, 2}, 0, 0, 1, 1, 1, 2, {5, 6}},
    {{7, 8}, {3, 2}, 0, 0, 1, 1, 2, 2, {3, 6}},
    {{7, 8}, {2, 8}, 0, 0, 1, 1, 6, 1, {}},
    {{7, 8}, {7, 2}, 0, 0, 1, 1, 1, 7, {}},
    {{7, 8}, {2, 2}, 0, 0, 1, 1, 6, 7, {}},
  };

  for (const auto &tc : test_cases) {
    EXPECT_EQ(
        Shape(tc.y),
        conv2d(
          tc.x, tc.w,
          tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1));
    EXPECT_EQ(
        Shape(tc.y, 3),
        conv2d(
          Shape(tc.x, 3), tc.w,
          tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1));
    EXPECT_EQ(
        Shape(tc.y, 3),
        conv2d(
          tc.x, Shape(tc.w, 3),
          tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1));
    EXPECT_EQ(
        Shape(tc.y, 3),
        conv2d(
          Shape(tc.x, 3), Shape(tc.w, 3),
          tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidConv2D) {
  struct TestCase {
    Shape x, w;
    std::uint32_t pad0, pad1, str0, str1, dil0, dil1;
    bool ok;
  };
  const vector<TestCase> test_cases {
    // invalid #dimensions
    {{1, 1, 1, 2}, {}, 0, 0, 1, 1, 1, 1, false},
    {{}, {1, 1, 1, 1, 2}, 0, 0, 1, 1, 1, 1, false},
    // zero-stride/dilation
    {{}, {}, 0, 0, 1, 1, 1, 1, true},
    {{}, {}, 0, 0, 0, 1, 1, 1, false},
    {{}, {}, 0, 0, 1, 0, 1, 1, false},
    {{}, {}, 0, 0, 1, 1, 0, 1, false},
    {{}, {}, 0, 0, 1, 1, 1, 0, false},
    // minibatches mismatching
    {Shape({}, 2), Shape({}, 2), 0, 0, 1, 1, 1, 1, true},
    {Shape({}, 3), Shape({}, 3), 0, 0, 1, 1, 1, 1, true},
    {Shape({}, 2), Shape({}, 3), 0, 0, 1, 1, 1, 1, false},
    // channels mismatching
    {{3, 3, 42}, {3, 3, 42}, 0, 0, 1, 1, 1, 1, true},
    {{3, 3, 42}, {3, 3, 43}, 0, 0, 1, 1, 1, 1, false},
    // sizes mismatching
    {{3, 3}, {3, 3}, 0, 0, 1, 1, 1, 1, true},
    {{3, 3}, {4, 3}, 0, 0, 1, 1, 1, 1, false},
    {{3, 3}, {3, 4}, 0, 0, 1, 1, 1, 1, false},
    {{3, 3}, {4, 4}, 0, 0, 1, 1, 1, 1, false},
    // sizes mismatching with padding
    {{3, 3}, {5, 5}, 1, 1, 1, 1, 1, 1, true},
    {{3, 3}, {6, 5}, 1, 1, 1, 1, 1, 1, false},
    {{3, 3}, {5, 6}, 1, 1, 1, 1, 1, 1, false},
    {{3, 3}, {6, 6}, 1, 1, 1, 1, 1, 1, false},
    // sizes mismatching with stride
    {{3, 3}, {3, 3}, 0, 0, 2, 2, 1, 1, true},
    {{3, 3}, {4, 3}, 0, 0, 2, 2, 1, 1, false},
    {{3, 3}, {3, 4}, 0, 0, 2, 2, 1, 1, false},
    {{3, 3}, {4, 4}, 0, 0, 2, 2, 1, 1, false},
    // sizes mismatching with dilation
    {{3, 3}, {2, 2}, 0, 0, 1, 1, 2, 2, true},
    {{2, 3}, {2, 2}, 0, 0, 1, 1, 2, 2, false},
    {{3, 2}, {2, 2}, 0, 0, 1, 1, 2, 2, false},
    {{2, 2}, {2, 2}, 0, 0, 1, 1, 2, 2, false},
    {{3, 3}, {2, 2}, 0, 0, 1, 1, 3, 2, false},
    {{3, 3}, {2, 2}, 0, 0, 1, 1, 2, 3, false},
    {{3, 3}, {2, 2}, 0, 0, 1, 1, 3, 3, false},
  };

  for (const auto &tc : test_cases) {
    if (tc.ok) {
      EXPECT_NO_THROW(
          conv2d(
            tc.x, tc.w, tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1));
    } else {
      EXPECT_THROW(
          conv2d(
            tc.x, tc.w, tc.pad0, tc.pad1, tc.str0, tc.str1, tc.dil0, tc.dil1),
          Error);
    }
  }
}

TEST_F(ShapeOpsTest, CheckPool2D) {
  struct TestCase {
    vector<std::uint32_t> x;
    std::uint32_t win0, win1, pad0, pad1, str0, str1;
    vector<std::uint32_t> y;
  };
  const vector<TestCase> test_cases {
    {{}, 1, 1, 0, 0, 1, 1, {}},
    {{7, 8}, 1, 1, 0, 0, 1, 1, {7, 8}},
    {{7, 8}, 3, 2, 0, 0, 1, 1, {5, 7}},
    {{7, 8}, 7, 8, 0, 0, 1, 1, {}},
    {{7, 8, 10}, 1, 1, 0, 0, 1, 1, {7, 8, 10}},
    {{7, 8, 10}, 3, 2, 0, 0, 1, 1, {5, 7, 10}},
    {{7, 8, 10}, 7, 8, 0, 0, 1, 1, {1, 1, 10}},
    // with padding
    {{7, 8}, 1, 1, 1, 0, 1, 1, {9, 8}},
    {{7, 8}, 1, 1, 0, 1, 1, 1, {7, 10}},
    {{7, 8}, 1, 1, 1, 1, 1, 1, {9, 10}},
    {{7, 8}, 3, 2, 1, 0, 1, 1, {7, 7}},
    {{7, 8}, 3, 2, 0, 1, 1, 1, {5, 9}},
    {{7, 8}, 3, 2, 1, 1, 1, 1, {7, 9}},
    {{7, 8}, 9, 8, 1, 0, 1, 1, {}},
    {{7, 8}, 7, 10, 0, 1, 1, 1, {}},
    {{7, 8}, 9, 10, 1, 1, 1, 1, {}},
    // with stride
    {{7, 8}, 1, 1, 0, 0, 2, 1, {4, 8}},
    {{7, 8}, 1, 1, 0, 0, 1, 2, {7, 4}},
    {{7, 8}, 1, 1, 0, 0, 2, 2, {4, 4}},
    {{7, 8}, 3, 2, 0, 0, 2, 1, {3, 7}},
    {{7, 8}, 3, 2, 0, 0, 1, 2, {5, 4}},
    {{7, 8}, 3, 2, 0, 0, 2, 2, {3, 4}},
    {{7, 8}, 7, 8, 0, 0, 2, 1, {}},
    {{7, 8}, 7, 8, 0, 0, 1, 2, {}},
    {{7, 8}, 7, 8, 0, 0, 2, 2, {}},
  };

  for (const auto &tc : test_cases) {
    EXPECT_EQ(
        Shape(tc.y),
        pool2d(tc.x, tc.win0, tc.win1, tc.pad0, tc.pad1, tc.str0, tc.str1));
    EXPECT_EQ(
        Shape(tc.y, 3),
        pool2d(
          Shape(tc.x, 3),
          tc.win0, tc.win1, tc.pad0, tc.pad1, tc.str0, tc.str1));
  }
}

TEST_F(ShapeOpsTest, CheckInvalidPool2D) {
  struct TestCase {
    Shape x;
    std::uint32_t win0, win1, pad0, pad1, str0, str1;
    bool ok;
  };
  const vector<TestCase> test_cases {
    // invalid #dimensions
    {{1, 1, 1, 2}, 1, 1, 0, 0, 1, 1, false},
    // zero-window/stride
    {{}, 1, 1, 0, 0, 1, 1, true},
    {{}, 0, 1, 0, 0, 1, 1, false},
    {{}, 1, 0, 0, 0, 1, 1, false},
    {{}, 1, 1, 0, 0, 0, 1, false},
    {{}, 1, 1, 0, 0, 1, 0, false},
    // sizes mismatching
    {{3, 3}, 3, 3, 0, 0, 1, 1, true},
    {{3, 3}, 4, 3, 0, 0, 1, 1, false},
    {{3, 3}, 3, 4, 0, 0, 1, 1, false},
    {{3, 3}, 4, 4, 0, 0, 1, 1, false},
    // sizes mismatching with padding
    {{3, 3}, 5, 5, 1, 1, 1, 1, true},
    {{3, 3}, 6, 5, 1, 1, 1, 1, false},
    {{3, 3}, 5, 6, 1, 1, 1, 1, false},
    {{3, 3}, 6, 6, 1, 1, 1, 1, false},
    // sizes mismatching with stride
    {{3, 3}, 3, 3, 0, 0, 2, 2, true},
    {{3, 3}, 4, 3, 0, 0, 2, 2, false},
    {{3, 3}, 3, 4, 0, 0, 2, 2, false},
    {{3, 3}, 4, 4, 0, 0, 2, 2, false},
  };

  for (const auto &tc : test_cases) {
    if (tc.ok) {
      EXPECT_NO_THROW(
          pool2d(tc.x, tc.win0, tc.win1, tc.pad0, tc.pad1, tc.str0, tc.str1));
    } else {
      EXPECT_THROW(
          pool2d(tc.x, tc.win0, tc.win1, tc.pad0, tc.pad1, tc.str0, tc.str1),
          Error);
    }
  }
}

TEST_F(ShapeOpsTest, CheckBatchPick) {
  struct TestCase {
    Shape input;
    vector<std::uint32_t> ids;
    Shape expected;
  };
  const vector<TestCase> test_cases {
    {{}, {0}, {}},
    {Shape({2, 2, 2}, 3), {0}, {2, 2, 2}},
    {Shape({2, 2, 2}, 3), {0, 0, 0}, Shape({2, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), {1, 0, 1}, Shape({2, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), {2, 1, 1}, Shape({2, 2, 2}, 3)},
    {Shape({2, 2, 2}, 3), {0, 1, 2, 1}, Shape({2, 2, 2}, 4)},
  };
  for (const TestCase &tc : test_cases) {
    const Shape observed = batch_pick(tc.input, tc.ids);
    EXPECT_EQ(tc.expected, observed);
  }
}

TEST_F(ShapeOpsTest, CheckInvalidBatchPick) {
  struct TestCase {
    Shape input;
    vector<std::uint32_t> ids;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2, 2}, 3), {}},
    {Shape({2, 2, 2}, 3), {3}},
    {Shape({2, 2, 2}, 3), {0, 1, 3}},
    {{}, {}},
    {{}, {1}},
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(batch_pick(tc.input, tc.ids), Error);
  }
}

TEST_F(ShapeOpsTest, CheckBatchSlice) {
  struct TestCase {
    std::uint32_t lower, upper;
    Shape input, expected;
  };
  const vector<TestCase> test_cases {
    {0, 1, {}, {}}, {0, 1, Shape({}, 2), {}},
    {1, 2, Shape({}, 2), {}}, {0, 2, Shape({}, 2), Shape({}, 2)},
    {0, 1, Shape({3}, 3), Shape({3}, 1)}, {1, 2, Shape({3}, 3), Shape({3}, 1)},
    {2, 3, Shape({3}, 3), Shape({3}, 1)}, {0, 2, Shape({3}, 3), Shape({3}, 2)},
    {1, 3, Shape({3}, 3), Shape({3}, 2)}, {0, 3, Shape({3}, 3), Shape({3}, 3)},
    {0, 1, Shape({2, 4}, 2), Shape({2, 4}, 1)},
    {1, 2, Shape({2, 4}, 2), Shape({2, 4}, 1)},
    {0, 2, Shape({2, 4}, 2), Shape({2, 4}, 2)},
  };
  for (const TestCase &tc : test_cases) {
    const Shape observed = batch_slice(tc.input, tc.lower, tc.upper);
    EXPECT_EQ(tc.expected, observed);
  }
}

TEST_F(ShapeOpsTest, CheckInvalidBatchSlice) {
  struct TestCase {
    std::uint32_t lower, upper;
    Shape input;
  };
  const vector<TestCase> test_cases {
    {0, 0, {}}, {1, 0, {}}, {0, 2, {}}, {1, 2, {}},
    {0, 0, Shape({}, 2)}, {1, 0, Shape({}, 2)}, {0, 3, Shape({}, 2)},
    {3, 1, Shape({}, 2)}, {2, 2, Shape({}, 2)}, {4, 3, Shape({}, 2)},
    {0, 0, {2}}, {0, 3, {2}}, {2, 1, {2}}, {0, 2, {2}},
    {1, 1, Shape({2, 2}, 3)}, {2, 1, Shape({2, 2}, 3)},
    {0, 4, Shape({2, 2}, 3)}, {3, 3, Shape({2, 3}, 3)}
  };
  for (const TestCase &tc : test_cases) {
    EXPECT_THROW(batch_slice(tc.input, tc.lower, tc.upper), Error);
  }
}

TEST_F(ShapeOpsTest, CheckBatchConcat) {
  struct TestCase {
    vector<Shape> inputs;
    Shape expected;
  };
  const vector<TestCase> test_cases {
    {{{}}, {}},
    {{{}, {}}, Shape({}, 2)},
    {{{}, {}, {}}, Shape({}, 3)},
    {{Shape({}, 2), {}, {}}, Shape({}, 4)},
    {{{}, Shape({}, 2), {}}, Shape({}, 4)},
    {{{}, {}, Shape({}, 2)}, Shape({}, 4)},
    {{Shape({}, 2), Shape({}, 3), Shape({}, 4)}, Shape({}, 9)},
    {{{5, 6, 7}}, {5, 6, 7}},
    {{{5, 6, 7}, {5, 6, 7}}, Shape({5, 6, 7}, 2)},
    {{{5, 6, 7}, {5, 6, 7}, {5, 6, 7}}, Shape({5, 6, 7}, 3)},
    {{Shape({5, 6, 7}, 2), {5, 6, 7}, {5, 6, 7}}, Shape({5, 6, 7}, 4)},
    {{{5, 6, 7}, Shape({5, 6, 7}, 2), {5, 6, 7}}, Shape({5, 6, 7}, 4)},
    {{{5, 6, 7}, {5, 6, 7}, Shape({5, 6, 7}, 2)}, Shape({5, 6, 7}, 4)},
    {{Shape({5, 6, 7}, 2), Shape({5, 6, 7}, 3), Shape({5, 6, 7}, 4)},
      Shape({5, 6, 7}, 9)},
  };
  for (const TestCase &tc : test_cases) {
    {
      // Using objects
      const Shape observed = batch_concat(tc.inputs);
      EXPECT_EQ(tc.expected, observed);
    }
    {
      // Using pointers
      vector<const Shape *> xs;
      for (const Shape &x : tc.inputs) xs.emplace_back(&x);
      const Shape observed = batch_concat(xs);
      EXPECT_EQ(tc.expected, observed);
    }
  }
}

TEST_F(ShapeOpsTest, CheckInvalidBatchConcat) {
  struct TestCase {
    vector<Shape> inputs;
    bool ok;
  };
  const vector<TestCase> test_cases {
    // empty
    {{}, false},
    // the last dim is invalid
    {{{}, {}}, true},
    {{{}, {2}}, false},
    {{{}, {1, 2}}, false},
    {{{}, {1, 1, 2}}, false},
    {{{}, {1, 1, 1, 2}}, false},
    // volume is correct but dims is invalid
    {{{2}, {2}}, true},
    {{{2}, {1, 2}}, false},
    {{{2}, {1, 1, 2}}, false},
    {{{2}, {1, 1, 1, 2}}, false},
    // only an axis is invalid
    {{{2, 3, 4}, {2, 3, 4}, {2, 3, 4}}, true},
    {{{3, 3, 4}, {2, 3, 4}, {2, 3, 4}}, false},
    {{{2, 4, 4}, {2, 3, 4}, {2, 3, 4}}, false},
    {{{2, 3, 5}, {2, 3, 4}, {2, 3, 4}}, false},
    {{{2, 3, 4}, {3, 3, 4}, {2, 3, 4}}, false},
    {{{2, 3, 4}, {2, 4, 4}, {2, 3, 4}}, false},
    {{{2, 3, 4}, {2, 3, 5}, {2, 3, 4}}, false},
    {{{2, 3, 4}, {2, 3, 4}, {3, 3, 4}}, false},
    {{{2, 3, 4}, {2, 3, 4}, {2, 4, 4}}, false},
    {{{2, 3, 4}, {2, 3, 4}, {2, 3, 5}}, false},
  };
  for (const TestCase &tc : test_cases) {
    // Using objects
    if (tc.ok) {
      EXPECT_NO_THROW(batch_concat(tc.inputs));
    } else {
      EXPECT_THROW(batch_concat(tc.inputs), Error);
    }
    // Using pointers
    vector<const Shape *>xs;
    for (const Shape &x : tc.inputs) xs.emplace_back(&x);
    if (tc.ok) {
      EXPECT_NO_THROW(batch_concat(xs));
    } else {
      EXPECT_THROW(batch_concat(xs), Error);
    }
  }
}

}  // namespace shape_ops
}  // namespace primitiv
