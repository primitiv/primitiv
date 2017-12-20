#include <primitiv/config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/c/shape.h>
#include <test_utils.h>

using std::vector;
using test_utils::array_match;

namespace primitiv {

class CShapeTest : public testing::Test {};

TEST_F(CShapeTest, CheckNewDefault) {
  {
    ::primitiv_Shape *shape = ::primitiv_Shape_new();
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
    std::uint32_t dims[] = {};
    ::primitiv_Shape *shape = ::primitiv_Shape_new_with_dims(dims, 0, 1);
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
    std::uint32_t dims[] = {1, 2, 3};
    ::primitiv_Shape *shape = ::primitiv_Shape_new_with_dims(dims, 3, 4);
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 0));
    EXPECT_EQ(2u, ::primitiv_Shape_op_getitem(shape, 1));
    EXPECT_EQ(3u, ::primitiv_Shape_op_getitem(shape, 2));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 3));
    EXPECT_EQ(1u, ::primitiv_Shape_op_getitem(shape, 100));
    std::uint32_t lhs[] = {1, 2, 3};
    std::uint32_t rhs[::primitiv_Shape_depth(shape)];
    ::primitiv_Shape_dims(shape, rhs);
    EXPECT_TRUE(array_match(lhs, rhs, 3));
    EXPECT_EQ(3u, ::primitiv_Shape_depth(shape));
    EXPECT_EQ(4u, ::primitiv_Shape_batch(shape));
    EXPECT_EQ(6u, ::primitiv_Shape_volume(shape));
    EXPECT_EQ(24u, ::primitiv_Shape_size(shape));
    ::primitiv_Shape_delete(shape);
  }
}

}  // namespace primitiv
