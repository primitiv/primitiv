#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>

using std::pair;
using std::string;
using std::vector;

namespace primitiv {

class ShapeTest : public testing::Test {};

TEST_F(ShapeTest, CheckNewDefault) {
  {
    const Shape shape;
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_EQ(0u, shape.dims().size());
    EXPECT_EQ(1u, shape.batch_size());
    EXPECT_EQ(1u, shape.size_per_sample());
    EXPECT_EQ(1u, shape.size());
  }
}

TEST_F(ShapeTest, CheckNewByInitializerList) {
  {
    const Shape shape({});
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_EQ(0u, shape.dims().size());
    EXPECT_EQ(1u, shape.batch_size());
    EXPECT_EQ(1u, shape.size_per_sample());
    EXPECT_EQ(1u, shape.size());
  }
  {
    const Shape shape({1, 2, 3}, 4);
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(2u, shape[1]);
    EXPECT_EQ(3u, shape[2]);
    EXPECT_EQ(1u, shape[3]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_EQ(3u, shape.dims().size());
    EXPECT_EQ(4u, shape.batch_size());
    EXPECT_EQ(6u, shape.size_per_sample());
    EXPECT_EQ(24u, shape.size());
  }
}

TEST_F(ShapeTest, CheckNewByVector) {
  {
    const Shape shape(vector<unsigned> {});
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_EQ(0u, shape.dims().size());
    EXPECT_EQ(1u, shape.batch_size());
    EXPECT_EQ(1u, shape.size_per_sample());
    EXPECT_EQ(1u, shape.size());
  }
  {
    const Shape shape(vector<unsigned> {1, 2, 3}, 4);
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(2u, shape[1]);
    EXPECT_EQ(3u, shape[2]);
    EXPECT_EQ(1u, shape[3]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_EQ(3u, shape.dims().size());
    EXPECT_EQ(4u, shape.batch_size());
    EXPECT_EQ(6u, shape.size_per_sample());
    EXPECT_EQ(24u, shape.size());
  }
}

TEST_F(ShapeTest, CheckInvalidNew) {
  EXPECT_THROW(Shape({0}), Error);
  EXPECT_THROW(Shape({2, 0}), Error);
  EXPECT_THROW(Shape({2, 3, 0}), Error);
  EXPECT_THROW(Shape({0}, 0), Error);
  EXPECT_THROW(Shape({2, 0}, 0), Error);
  EXPECT_THROW(Shape({2, 3, 0}, 0), Error);
  EXPECT_THROW(Shape({}, 0), Error);
}

TEST_F(ShapeTest, CheckString) {
  vector<pair<Shape, string>> cases {
    {Shape(), "[]x1"},
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
    const Shape target({1, 1}, 1);
    const vector<Shape> eq {
      Shape(),
      Shape({}),
      Shape({}, 1),
      Shape({1}),
      Shape({1}, 1),
      Shape({1, 1}),
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
      Shape(),
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
  const Shape src1({2, 3, 5}, 7);
  const Shape src2({1, 4}, 9);

  // c-tor
  Shape copied = src1;
  EXPECT_EQ(src1, copied);

  // operator=
  copied = src2;
  EXPECT_EQ(src2, copied);
}

TEST_F(ShapeTest, CheckMove) {
  Shape src1({2, 3, 5}, 7);
  const Shape trg1({2, 3, 5}, 7);
  Shape src2({1, 4}, 9);
  const Shape trg2({1, 4}, 9);

  // c-tor
  Shape moved = std::move(src1);
  EXPECT_EQ(trg1, moved);

  // operator=
  moved = std::move(src2);
  EXPECT_EQ(trg2, moved);
}

}  // namespace primitiv
