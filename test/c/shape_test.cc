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
    ASSERT_EQ(::primitiv_Status::PRIMITIV_OK,
              ::primitiv_Shape_new(&shape));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 0));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 1));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 100));
    EXPECT_EQ(0u, ::primitiv_Shape_depth(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_batch(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_volume(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_size(shape));
    ::primitiv_Shape_delete(shape);
  }
}

TEST_F(CShapeTest, CheckNewByArray) {
  {
    ::primitiv_Shape *shape;
    std::uint32_t dims[] = {};
    ASSERT_EQ(::primitiv_Status::PRIMITIV_OK,
              ::primitiv_Shape_new_with_dims(dims, 0, 1, &shape));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 0));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 1));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 100));
    EXPECT_EQ(0u, ::primitiv_Shape_depth(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_batch(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_volume(shape));
    EXPECT_EQ(1u, ::primitiv_Shape_size(shape));
    ::primitiv_Shape_delete(shape);
  }
  {
    ::primitiv_Shape *shape;
    std::uint32_t dims[] = {1, 2, 3};
    ASSERT_EQ(::primitiv_Status::PRIMITIV_OK,
              ::primitiv_Shape_new_with_dims(dims, 3, 4, &shape));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 0));
    EXPECT_EQ(2u, ::primitiv_Shape_op_getitem(shape, 1));
    EXPECT_EQ(3u, ::primitiv_Shape_op_getitem(shape, 2));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 3));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 100));
    std::uint32_t lhs[] = {1, 2, 3};
    std::size_t rhs_size;
    ::primitiv_Shape_dims(shape, nullptr, &rhs_size);
    std::uint32_t rhs[rhs_size];
    ::primitiv_Shape_dims(shape, rhs, &rhs_size);
    EXPECT_TRUE(array_match(lhs, rhs, 3));
    EXPECT_EQ(3u, ::primitiv_Shape_depth(shape));
    EXPECT_EQ(4u, ::primitiv_Shape_batch(shape));
    EXPECT_EQ(6u, ::primitiv_Shape_volume(shape));
    EXPECT_EQ(24u, ::primitiv_Shape_size(shape));
    ::primitiv_Shape_delete(shape);
  }
}

TEST_F(CShapeTest, CheckInvalidNew) {
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 1, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 1, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 2, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 3, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 0, 0, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_OK,
              ::primitiv_Shape_new_with_dims(dims, 8, 10, &shape));
  }
  {
    ::primitiv_Shape* shape;
    std::uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
              ::primitiv_Shape_new_with_dims(dims, 9, 10, &shape));
  }
}

TEST_F(CShapeTest, CheckNumElementsUnderRank) {
  ::primitiv_Shape *src;
  std::uint32_t dims[] = {2, 3, 5, 7, 11, 13};
  EXPECT_EQ(::primitiv_Status::PRIMITIV_OK,
            ::primitiv_Shape_new_with_dims(dims, 6, 17, &src));
  EXPECT_EQ(1u, ::primitiv_Shape_lower_volume(src, 0));
  EXPECT_EQ(2u, ::primitiv_Shape_lower_volume(src, 1));
  EXPECT_EQ(2u * 3u, ::primitiv_Shape_lower_volume(src, 2));
  EXPECT_EQ(2u * 3u * 5u, ::primitiv_Shape_lower_volume(src, 3));
  EXPECT_EQ(2u * 3u * 5u * 7u, ::primitiv_Shape_lower_volume(src, 4));
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u, ::primitiv_Shape_lower_volume(src, 5));
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u * 13u,
      ::primitiv_Shape_lower_volume(src, 6));
}

}  // namespace c
}  // namespace primitiv
