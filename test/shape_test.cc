#include <primitiv/config.h>

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/core/shape.h>

#include <test_utils.h>

using std::pair;
using std::string;
using std::vector;
using test_utils::vector_match;

namespace primitiv {

class ShapeTest : public testing::Test {};

TEST_F(ShapeTest, CheckNewDefault) {
  {
    const Shape shape;
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_TRUE(vector_match(vector<std::uint32_t> {}, shape.dims()));
    EXPECT_EQ(0u, shape.depth());
    EXPECT_EQ(1u, shape.batch());
    EXPECT_EQ(1u, shape.volume());
    EXPECT_EQ(1u, shape.size());
  }
}

TEST_F(ShapeTest, CheckNewByInitializerList) {
  {
    const Shape shape({});
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_TRUE(vector_match(vector<std::uint32_t> {}, shape.dims()));
    EXPECT_EQ(0u, shape.depth());
    EXPECT_EQ(1u, shape.batch());
    EXPECT_EQ(1u, shape.volume());
    EXPECT_EQ(1u, shape.size());
  }
  {
    const Shape shape({1, 2, 3}, 4);
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(2u, shape[1]);
    EXPECT_EQ(3u, shape[2]);
    EXPECT_EQ(1u, shape[3]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_TRUE(vector_match(vector<std::uint32_t> { 1, 2, 3 }, shape.dims()));
    EXPECT_EQ(3u, shape.depth());
    EXPECT_EQ(4u, shape.batch());
    EXPECT_EQ(6u, shape.volume());
    EXPECT_EQ(24u, shape.size());
  }
}

TEST_F(ShapeTest, CheckNewByVector) {
  {
    const Shape shape(vector<std::uint32_t> {});
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(1u, shape[1]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_TRUE(vector_match(vector<std::uint32_t> {}, shape.dims()));
    EXPECT_EQ(0u, shape.depth());
    EXPECT_EQ(1u, shape.batch());
    EXPECT_EQ(1u, shape.volume());
    EXPECT_EQ(1u, shape.size());
  }
  {
    const Shape shape(vector<std::uint32_t> {1, 2, 3}, 4);
    EXPECT_EQ(1u, shape[0]);
    EXPECT_EQ(2u, shape[1]);
    EXPECT_EQ(3u, shape[2]);
    EXPECT_EQ(1u, shape[3]);
    EXPECT_EQ(1u, shape[100]);
    EXPECT_TRUE(vector_match(vector<std::uint32_t> { 1, 2, 3 }, shape.dims()));
    EXPECT_EQ(3u, shape.depth());
    EXPECT_EQ(4u, shape.batch());
    EXPECT_EQ(6u, shape.volume());
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
  EXPECT_NO_THROW(Shape({1, 2, 3, 4, 5, 6, 7, 8}, 10));
  EXPECT_THROW(Shape({1, 2, 3, 4, 5, 6, 7, 8, 9}, 10), Error);
  EXPECT_NO_THROW(Shape(vector<std::uint32_t> {1, 2, 3, 4, 5, 6, 7, 8}, 10));
  EXPECT_THROW(Shape(vector<std::uint32_t> {1, 2, 3, 4, 5, 6, 7, 8, 9}, 10), Error);
}

TEST_F(ShapeTest, CheckNumElementsUnderRank) {
  Shape src({2, 3, 5, 7, 11, 13}, 17);
  EXPECT_EQ(1u, src.lower_volume(0));
  EXPECT_EQ(2u, src.lower_volume(1));
  EXPECT_EQ(2u * 3u, src.lower_volume(2));
  EXPECT_EQ(2u * 3u * 5u, src.lower_volume(3));
  EXPECT_EQ(2u * 3u * 5u * 7u, src.lower_volume(4));
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u, src.lower_volume(5));
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u * 13u, src.lower_volume(6));
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

TEST_F(ShapeTest, CheckCopyToThis) {
  Shape a({2, 3, 5}, 7);
  a = a;
  EXPECT_EQ(Shape({2, 3, 5}, 7), a);
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

#if 0
// Some compilers does not compile this test due to "-Wself-move".
TEST_F(ShapeTest, CheckMoveToThis) {
  Shape a({2, 3, 5}, 7);
  a = std::move(a);
  EXPECT_EQ(Shape({2, 3, 5}, 7), a);
}
#endif

TEST_F(ShapeTest, CheckHasBatch) {
  EXPECT_FALSE(Shape().has_batch());
  EXPECT_FALSE(Shape({2}).has_batch());
  EXPECT_FALSE(Shape({2, 3}).has_batch());
  EXPECT_FALSE(Shape({2, 3, 4}).has_batch());
  EXPECT_TRUE(Shape({}, 5).has_batch());
  EXPECT_TRUE(Shape({2}, 5).has_batch());
  EXPECT_TRUE(Shape({2, 3}, 5).has_batch());
  EXPECT_TRUE(Shape({2, 3, 4}, 5).has_batch());
}

TEST_F(ShapeTest, CheckHasCompatibleBatch) {
  const Shape src1({2, 3, 5});
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 5})));
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 5}, 2)));
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 5}, 7)));
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 4})));
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 4}, 2)));
  EXPECT_TRUE(src1.has_compatible_batch(Shape({2, 3, 4}, 7)));

  const Shape src2({2, 3, 5}, 7);
  EXPECT_TRUE(src2.has_compatible_batch(Shape({2, 3, 5})));
  EXPECT_FALSE(src2.has_compatible_batch(Shape({2, 3, 5}, 2)));
  EXPECT_TRUE(src2.has_compatible_batch(Shape({2, 3, 5}, 7)));
  EXPECT_TRUE(src2.has_compatible_batch(Shape({2, 3, 4})));
  EXPECT_FALSE(src2.has_compatible_batch(Shape({2, 3, 4}, 2)));
  EXPECT_TRUE(src2.has_compatible_batch(Shape({2, 3, 4}, 7)));
}

TEST_F(ShapeTest, CheckIsScalar) {
  EXPECT_TRUE(Shape().is_scalar());
  EXPECT_FALSE(Shape({2}).is_scalar());
  EXPECT_FALSE(Shape({2, 3}).is_scalar());
  EXPECT_FALSE(Shape({2, 3, 4}).is_scalar());
  EXPECT_TRUE(Shape({}, 5).is_scalar());
  EXPECT_FALSE(Shape({2}, 5).is_scalar());
  EXPECT_FALSE(Shape({2, 3}, 5).is_scalar());
  EXPECT_FALSE(Shape({2, 3, 4}, 5).is_scalar());
}

TEST_F(ShapeTest, CheckIsColumnVector) {
  EXPECT_TRUE(Shape().is_column_vector());
  EXPECT_TRUE(Shape({2}).is_column_vector());
  EXPECT_FALSE(Shape({2, 3}).is_column_vector());
  EXPECT_FALSE(Shape({2, 3, 4}).is_column_vector());
  EXPECT_TRUE(Shape({}, 5).is_column_vector());
  EXPECT_TRUE(Shape({2}, 5).is_column_vector());
  EXPECT_FALSE(Shape({2, 3}, 5).is_column_vector());
  EXPECT_FALSE(Shape({2, 3, 4}, 5).is_column_vector());
}

TEST_F(ShapeTest, CheckIsMatrix) {
  EXPECT_TRUE(Shape().is_matrix());
  EXPECT_TRUE(Shape({2}).is_matrix());
  EXPECT_TRUE(Shape({2, 3}).is_matrix());
  EXPECT_FALSE(Shape({2, 3, 4}).is_matrix());
  EXPECT_TRUE(Shape({}, 5).is_matrix());
  EXPECT_TRUE(Shape({2}, 5).is_matrix());
  EXPECT_TRUE(Shape({2, 3}, 5).is_matrix());
  EXPECT_FALSE(Shape({2, 3, 4}, 5).is_matrix());
}

TEST_F(ShapeTest, CheckHasSameDims) {
  const Shape src1({2, 3, 5});
  EXPECT_TRUE(src1.has_same_dims(Shape({2, 3, 5})));
  EXPECT_TRUE(src1.has_same_dims(Shape({2, 3, 5}, 2)));
  EXPECT_TRUE(src1.has_same_dims(Shape({2, 3, 5}, 7)));
  EXPECT_FALSE(src1.has_same_dims(Shape({2, 3, 4})));
  EXPECT_FALSE(src1.has_same_dims(Shape({2, 3, 4}, 2)));
  EXPECT_FALSE(src1.has_same_dims(Shape({2, 3, 4}, 7)));

  const Shape src2({2, 3, 5}, 7);
  EXPECT_TRUE(src2.has_same_dims(Shape({2, 3, 5})));
  EXPECT_TRUE(src2.has_same_dims(Shape({2, 3, 5}, 2)));
  EXPECT_TRUE(src2.has_same_dims(Shape({2, 3, 5}, 7)));
  EXPECT_FALSE(src2.has_same_dims(Shape({2, 3, 4})));
  EXPECT_FALSE(src2.has_same_dims(Shape({2, 3, 4}, 2)));
  EXPECT_FALSE(src2.has_same_dims(Shape({2, 3, 4}, 7)));
}

TEST_F(ShapeTest, CheckHasSameDimsLOO) {
  const Shape src({2, 3, 5});
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5}, 0));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5}, 1));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5}, 2));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5}, 3));
  EXPECT_TRUE(src.has_same_loo_dims({10, 3, 5}, 0));
  EXPECT_TRUE(src.has_same_loo_dims({2, 10, 5}, 1));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 10}, 2));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5}, 3));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5, 10}, 3));
  EXPECT_TRUE(src.has_same_loo_dims({2, 3, 5, 1, 10}, 4));
  EXPECT_FALSE(src.has_same_loo_dims({10, 3, 5}, 1));
  EXPECT_FALSE(src.has_same_loo_dims({2, 10, 5}, 0));
  EXPECT_FALSE(src.has_same_loo_dims({2, 3, 10}, 0));
  EXPECT_FALSE(src.has_same_loo_dims({2, 3, 5, 10}, 0));
  EXPECT_FALSE(src.has_same_loo_dims({2, 3, 5, 10, 10}, 3));
  EXPECT_FALSE(src.has_same_loo_dims({20, 30, 50}, 0));
}

TEST_F(ShapeTest, CheckResizeDim) {
  const Shape src({2, 3, 5}, 7);

  EXPECT_EQ(Shape({1, 3, 5}, 7), src.resize_dim(0, 1));
  EXPECT_EQ(105u, src.resize_dim(0, 1).size());
  EXPECT_EQ(Shape({10, 3, 5}, 7), src.resize_dim(0, 10));
  EXPECT_EQ(1050u, src.resize_dim(0, 10).size());

  EXPECT_EQ(Shape({2, 1, 5}, 7), src.resize_dim(1, 1));
  EXPECT_EQ(70u, src.resize_dim(1, 1).size());
  EXPECT_EQ(Shape({2, 10, 5}, 7), src.resize_dim(1, 10));
  EXPECT_EQ(700u, src.resize_dim(1, 10).size());

  EXPECT_EQ(Shape({2, 3, 1}, 7), src.resize_dim(2, 1));
  EXPECT_EQ(42u, src.resize_dim(2, 1).size());
  EXPECT_EQ(Shape({2, 3, 10}, 7), src.resize_dim(2, 10));
  EXPECT_EQ(420u, src.resize_dim(2, 10).size());

  EXPECT_EQ(Shape({2, 3, 5, 10}, 7), src.resize_dim(3, 10));
  EXPECT_EQ(2100u, src.resize_dim(3, 10).size());

  EXPECT_EQ(Shape({2, 3, 5, 1, 10}, 7), src.resize_dim(4, 10));
  EXPECT_EQ(2100u, src.resize_dim(4, 10).size());

  EXPECT_THROW(src.resize_dim(0, 0), Error);
  EXPECT_NO_THROW(src.resize_dim(Shape::MAX_DEPTH - 1, 1));
  EXPECT_THROW(src.resize_dim(Shape::MAX_DEPTH, 1), Error);
}

TEST_F(ShapeTest, CheckResizeBatch) {
  const Shape src({2, 3, 5}, 7);

  EXPECT_EQ(Shape({2, 3, 5}), src.resize_batch(1));
  EXPECT_EQ(30u, src.resize_batch(1).size());

  EXPECT_EQ(Shape({2, 3, 5}, 2), src.resize_batch(2));
  EXPECT_EQ(60u, src.resize_batch(2).size());

  EXPECT_EQ(Shape({2, 3, 5}, 4), src.resize_batch(4));
  EXPECT_EQ(120u, src.resize_batch(4).size());

  EXPECT_THROW(src.resize_batch(0), Error);
}

TEST_F(ShapeTest, CheckUpdateDim) {
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(0, 1);
    EXPECT_EQ(Shape({1, 3, 5}, 7), src);
    EXPECT_EQ(105u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(0, 10);
    EXPECT_EQ(Shape({10, 3, 5}, 7), src);
    EXPECT_EQ(1050u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(1, 1);
    EXPECT_EQ(Shape({2, 1, 5}, 7), src);
    EXPECT_EQ(70u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(1, 10);
    EXPECT_EQ(Shape({2, 10, 5}, 7), src);
    EXPECT_EQ(700u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(2, 1);
    EXPECT_EQ(Shape({2, 3, 1}, 7), src);
    EXPECT_EQ(42u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(2, 10);
    EXPECT_EQ(Shape({2, 3, 10}, 7), src);
    EXPECT_EQ(420u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(3, 10);
    EXPECT_EQ(Shape({2, 3, 5, 10}, 7), src);
    EXPECT_EQ(2100u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    src.update_dim(4, 10);
    EXPECT_EQ(Shape({2, 3, 5, 1, 10}, 7), src);
    EXPECT_EQ(2100u, src.size());
  }
  {
    Shape src({2, 3, 5}, 7);
    EXPECT_THROW(src.update_dim(0, 0), Error);
  }
  {
    Shape src({2, 3, 5}, 7);
    EXPECT_NO_THROW(src.update_dim(Shape::MAX_DEPTH - 1, 1));
    EXPECT_THROW(src.update_dim(Shape::MAX_DEPTH, 1), Error);
  }
}

TEST_F(ShapeTest, CheckUpdateBatch) {
  Shape src({2, 3, 5}, 7);
  src.update_batch(1);
  EXPECT_EQ(Shape({2, 3, 5}), src);
  EXPECT_EQ(30u, src.size());
  src.update_batch(2);
  EXPECT_EQ(Shape({2, 3, 5}, 2), src);
  EXPECT_EQ(60u, src.size());
  src.update_batch(4);
  EXPECT_EQ(Shape({2, 3, 5}, 4), src);
  EXPECT_EQ(120u, src.size());
  EXPECT_THROW(src.update_batch(0), Error);
}

}  // namespace primitiv
