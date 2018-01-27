#include <primitiv/config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/c/shape.h>
#include <test_utils.h>

using std::vector;
using test_utils::array_match;

namespace primitiv {
namespace c {

class CShapeTest : public testing::Test {};

TEST_F(CShapeTest, CheckNewDefault) {
  ::primitivShape_t *shape;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitiv_Shape_new(&shape));

  uint32_t ret = 0u;
  ::primitiv_Shape_at(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 1, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  ret = 1u;
  ::primitiv_Shape_depth(shape, &ret);
  EXPECT_EQ(0u, ret);

  ret = 0u;
  ::primitiv_Shape_batch(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_volume(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_size(shape, &ret);
  EXPECT_EQ(1u, ret);

  ::primitiv_Shape_delete(shape);
}

TEST_F(CShapeTest, CheckNewByArray1) {
  ::primitivShape_t *shape;
  uint32_t dims[] = {};
  ASSERT_EQ(PRIMITIV_C_OK, ::primitiv_Shape_new_with_dims(dims, 0, 1, &shape));

  uint32_t ret = 0u;
  ::primitiv_Shape_at(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 1, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  ret = 1u;
  ::primitiv_Shape_depth(shape, &ret);
  EXPECT_EQ(0u, ret);

  ret = 0u;
  ::primitiv_Shape_batch(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_volume(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_size(shape, &ret);
  EXPECT_EQ(1u, ret);

  ::primitiv_Shape_delete(shape);
}

TEST_F(CShapeTest, CheckNewByArray2) {
  ::primitivShape_t *shape;
  uint32_t dims[] = {1, 2, 3};
  ASSERT_EQ(PRIMITIV_C_OK, ::primitiv_Shape_new_with_dims(dims, 3, 4, &shape));

  uint32_t ret = 0u;
  ::primitiv_Shape_at(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 1, &ret);
  EXPECT_EQ(2u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 2, &ret);
  EXPECT_EQ(3u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 3, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitiv_Shape_at(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  size_t dims_size = 0u;
  ::primitiv_Shape_dims(shape, nullptr, &dims_size);
  EXPECT_EQ(3u, dims_size);

  uint32_t rhs[dims_size];
  ::primitiv_Shape_dims(shape, rhs, &dims_size);
  EXPECT_TRUE(array_match(dims, rhs, dims_size));

  ret = 0u;
  ::primitiv_Shape_depth(shape, &ret);
  EXPECT_EQ(3u, ret);

  ret = 0u;
  ::primitiv_Shape_batch(shape, &ret);
  EXPECT_EQ(4u, ret);

  ret = 0u;
  ::primitiv_Shape_volume(shape, &ret);
  EXPECT_EQ(6u, ret);

  ret = 0u;
  ::primitiv_Shape_size(shape, &ret);
  EXPECT_EQ(24u, ret);

  ::primitiv_Shape_delete(shape);
}

TEST_F(CShapeTest, CheckInvalidNew) {
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 0, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(PRIMITIV_C_OK,
              ::primitiv_Shape_new_with_dims(dims, 8, 10, &shape));
  }
  {
    ::primitiv_Shape* shape;
    uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 9, 10, &shape));
  }
}

TEST_F(CShapeTest, CheckNumElementsUnderRank) {
  ::primitivShape_t *src;
  uint32_t dims[] = {2, 3, 5, 7, 11, 13};
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Shape_new_with_dims(dims, 6, 17, &src));
  uint32_t lower_volume;
  ::primitiv_Shape_lower_volume(src, 0, &lower_volume);
  EXPECT_EQ(1u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 1, &lower_volume);
  EXPECT_EQ(2u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 2, &lower_volume);
  EXPECT_EQ(2u * 3u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 3, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 4, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 5, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u, lower_volume);
  ::primitiv_Shape_lower_volume(src, 6, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u * 13u, lower_volume);
}

}  // namespace c
}  // namespace primitiv
