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
  {
    ::primitiv_Shape *shape;
    ASSERT_EQ(PRIMITIV_C_OK,
              ::primitiv_Shape_new(&shape));
    uint32_t dim_size;
    ::primitiv_Shape_op_getitem(shape, 0, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 1, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 100, &dim_size);
    EXPECT_EQ(1u, dim_size);
    uint32_t depth;
    ::primitiv_Shape_depth(shape, &depth);
    EXPECT_EQ(0u, depth);
    uint32_t batch;
    ::primitiv_Shape_batch(shape, &batch);
    EXPECT_EQ(1u, batch);
    uint32_t volume;
    ::primitiv_Shape_volume(shape, &volume);
    EXPECT_EQ(1u, volume);
    uint32_t size;
    ::primitiv_Shape_size(shape, &size);
    EXPECT_EQ(1u, size);
    ::primitiv_Shape_delete(shape);
  }
}

TEST_F(CShapeTest, CheckNewByArray) {
  {
    ::primitiv_Shape *shape;
    std::uint32_t dims[] = {};
    ASSERT_EQ(PRIMITIV_C_OK,
              ::primitiv_Shape_new_with_dims(dims, 0, 1, &shape));
    uint32_t dim_size;
    ::primitiv_Shape_op_getitem(shape, 0, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 1, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 100, &dim_size);
    EXPECT_EQ(1u, dim_size);
    uint32_t depth;
    ::primitiv_Shape_depth(shape, &depth);
    EXPECT_EQ(0u, depth);
    uint32_t batch;
    ::primitiv_Shape_batch(shape, &batch);
    EXPECT_EQ(1u, batch);
    uint32_t volume;
    ::primitiv_Shape_volume(shape, &volume);
    EXPECT_EQ(1u, volume);
    uint32_t size;
    ::primitiv_Shape_size(shape, &size);
    EXPECT_EQ(1u, size);
    ::primitiv_Shape_delete(shape);
  }
  {
    ::primitiv_Shape *shape;
    std::uint32_t dims[] = {1, 2, 3};
    ASSERT_EQ(PRIMITIV_C_OK,
              ::primitiv_Shape_new_with_dims(dims, 3, 4, &shape));
    uint32_t dim_size;
    ::primitiv_Shape_op_getitem(shape, 0, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 1, &dim_size);
    EXPECT_EQ(2u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 2, &dim_size);
    EXPECT_EQ(3u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 3, &dim_size);
    EXPECT_EQ(1u, dim_size);
    ::primitiv_Shape_op_getitem(shape, 100, &dim_size);
    EXPECT_EQ(1u, dim_size);
    std::uint32_t lhs[] = {1, 2, 3};
    std::size_t rhs_size;
    ::primitiv_Shape_dims(shape, nullptr, &rhs_size);
    std::uint32_t rhs[rhs_size];
    ::primitiv_Shape_dims(shape, rhs, &rhs_size);
    EXPECT_TRUE(array_match(lhs, rhs, 3));
    uint32_t depth;
    ::primitiv_Shape_depth(shape, &depth);
    EXPECT_EQ(3u, depth);
    uint32_t batch;
    ::primitiv_Shape_batch(shape, &batch);
    EXPECT_EQ(4u, batch);
    uint32_t volume;
    ::primitiv_Shape_volume(shape, &volume);
    EXPECT_EQ(6u, volume);
    uint32_t size;
    ::primitiv_Shape_size(shape, &size);
    EXPECT_EQ(24u, size);
    ::primitiv_Shape_delete(shape);
  }
}

TEST_F(CShapeTest, CheckInvalidNew) {
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 0, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(PRIMITIV_C_OK,
              ::primitiv_Shape_new_with_dims(dims, 8, 10, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 9, 10, &shape));
  }
}

TEST_F(CShapeTest, CheckNumElementsUnderRank) {
  ::primitiv_Shape *src;
  std::uint32_t dims[] = {2, 3, 5, 7, 11, 13};
  EXPECT_EQ(PRIMITIV_C_OK,
            ::primitiv_Shape_new_with_dims(dims, 6, 17, &src));
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
