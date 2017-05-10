#include <config.h>

#include <stdexcept>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/shape.h>

using std::pair;
using std::string;
using std::vector;

namespace primitiv {

class ShapeTest : public testing::Test {};

TEST_F(ShapeTest, CheckNew) {
  {
    const Shape shape({});
    EXPECT_EQ(1u, shape.dim(0));
    EXPECT_EQ(1u, shape.dim(1));
    EXPECT_EQ(1u, shape.dim(100));
    EXPECT_EQ(0u, shape.dim_size());
    EXPECT_EQ(1u, shape.batch_size());
    EXPECT_EQ(1u, shape.size());
  }
  {
    const Shape shape({1, 2, 3}, 4);
    EXPECT_EQ(1u, shape.dim(0));
    EXPECT_EQ(2u, shape.dim(1));
    EXPECT_EQ(3u, shape.dim(2));
    EXPECT_EQ(1u, shape.dim(3));
    EXPECT_EQ(1u, shape.dim(100));
    EXPECT_EQ(3u, shape.dim_size());
    EXPECT_EQ(4u, shape.batch_size());
    EXPECT_EQ(24u, shape.size());
  }
}

TEST_F(ShapeTest, CheckInvalidNew) {
  EXPECT_THROW(Shape({0}), std::runtime_error);
  EXPECT_THROW(Shape({2, 0}), std::runtime_error);
  EXPECT_THROW(Shape({2, 3, 0}), std::runtime_error);
  EXPECT_THROW(Shape({0}, 0), std::runtime_error);
  EXPECT_THROW(Shape({2, 0}, 0), std::runtime_error);
  EXPECT_THROW(Shape({2, 3, 0}, 0), std::runtime_error);
  EXPECT_THROW(Shape({}, 0), std::runtime_error);
}

TEST_F(ShapeTest, CheckString) {
  vector<pair<Shape, string>> cases {
    {Shape({}), "[]x1"},
    {Shape({1}), "[]x1"},
    {Shape({1, 1}), "[]x1"},
    {Shape({}, 1), "[]x1"},
    {Shape({1}, 1), "[]x1"},
    {Shape({1, 1}, 1), "[]x1"},
    {Shape({}, 2), "[]x2"},
    {Shape({1}, 2), "[]x2"},
    {Shape({1, 1}, 2), "[]x2"},
    {Shape({2}), "[2]x1"},
    {Shape({2, 1}), "[2]x1"},
    {Shape({2, 3}), "[2,3]x1"},
    {Shape({2, 3, 1}), "[2,3]x1"},
    {Shape({2, 3, 5}), "[2,3,5]x1"},
    {Shape({2, 3, 5, 1}), "[2,3,5]x1"},
    {Shape({2}, 3), "[2]x3"},
    {Shape({2, 1}, 3), "[2]x3"},
    {Shape({2, 3}, 5), "[2,3]x5"},
    {Shape({2, 3, 1}, 5), "[2,3]x5"},
    {Shape({2, 3, 5}, 7), "[2,3,5]x7"},
    {Shape({2, 3, 5, 1}, 7), "[2,3,5]x7"},
  };
  for (const auto & c : cases) {
    EXPECT_EQ(c.second, c.first.to_string());
  }
}

TEST_F(ShapeTest, CheckCmp) {
  {
    const Shape target({});
    const vector<Shape> eq {
      Shape({}, 1),
      Shape({1}),
      Shape({1}, 1),
      Shape({1, 1}),
      Shape({1, 1}, 1)
    };
    const vector<Shape> ne {
      Shape({}, 2),
      Shape({2}),
      Shape({2}, 2),
      Shape({1, 2}),
      Shape({1, 2}, 2),
    };
    for (const Shape & s : eq) {
      EXPECT_EQ(target, s);
    }
    for (const Shape & s : ne) {
      EXPECT_NE(target, s);
    }
  }
  {
    const Shape target({2, 3}, 5);
    const vector<Shape> eq {
      Shape({2, 3}, 5),
      Shape({2, 3, 1}, 5),
    };
    const vector<Shape> ne {
      Shape({}),
      Shape({2}),
      Shape({2, 3}),
      Shape({3, 2}),
      Shape({}, 5),
      Shape({2}, 5),
      Shape({3, 2}, 5),
    };
    for (const Shape & s : eq) {
      EXPECT_EQ(target, s);
    }
    for (const Shape & s : ne) {
      EXPECT_NE(target, s);
    }
  }
}

TEST_F(ShapeTest, CheckCopy) {
  const Shape src({2, 3, 5}, 7);
  {
    const Shape copied = src;
    EXPECT_EQ(src, copied);
    Shape temp = src;
    const Shape moved = std::move(temp);
    EXPECT_EQ(src, moved);
  }
  {
    Shape copied({});
    copied = src;
    EXPECT_EQ(src, copied);
    Shape temp = src;
    Shape moved({});
    moved = std::move(temp);
    EXPECT_EQ(src, moved);
  }
}

}  // namespace primitiv
