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
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateShape(&shape));

  uint32_t ret = 0u;
  ::primitivGetShapeDimSize(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 1, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  ret = 1u;
  ::primitivGetShapeDepth(shape, &ret);
  EXPECT_EQ(0u, ret);

  ret = 0u;
  ::primitivGetShapeBatchSize(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeVolume(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeSize(shape, &ret);
  EXPECT_EQ(1u, ret);

  ::primitivDeleteShape(shape);
}

TEST_F(CShapeTest, CheckNewByArray1) {
  ::primitivShape_t *shape;
  uint32_t dims[] = {};
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateShapeWithDims(dims, 0, 1, &shape));

  uint32_t ret = 0u;
  ::primitivGetShapeDimSize(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 1, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  ret = 1u;
  ::primitivGetShapeDepth(shape, &ret);
  EXPECT_EQ(0u, ret);

  ret = 0u;
  ::primitivGetShapeBatchSize(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeVolume(shape, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeSize(shape, &ret);
  EXPECT_EQ(1u, ret);

  ::primitivDeleteShape(shape);
}

TEST_F(CShapeTest, CheckNewByArray2) {
  ::primitivShape_t *shape;
  uint32_t dims[] = {1, 2, 3};
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateShapeWithDims(dims, 3, 4, &shape));

  uint32_t ret = 0u;
  ::primitivGetShapeDimSize(shape, 0, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 1, &ret);
  EXPECT_EQ(2u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 2, &ret);
  EXPECT_EQ(3u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 3, &ret);
  EXPECT_EQ(1u, ret);

  ret = 0u;
  ::primitivGetShapeDimSize(shape, 100, &ret);
  EXPECT_EQ(1u, ret);

  size_t dims_size = 0u;
  ::primitivGetShapeDims(shape, nullptr, &dims_size);
  EXPECT_EQ(3u, dims_size);

  uint32_t rhs[dims_size];
  ::primitivGetShapeDims(shape, rhs, &dims_size);
  EXPECT_TRUE(array_match(dims, rhs, dims_size));

  ret = 0u;
  ::primitivGetShapeDepth(shape, &ret);
  EXPECT_EQ(3u, ret);

  ret = 0u;
  ::primitivGetShapeBatchSize(shape, &ret);
  EXPECT_EQ(4u, ret);

  ret = 0u;
  ::primitivGetShapeVolume(shape, &ret);
  EXPECT_EQ(6u, ret);

  ret = 0u;
  ::primitivGetShapeSize(shape, &ret);
  EXPECT_EQ(24u, ret);

  ::primitivDeleteShape(shape);
}

TEST_F(CShapeTest, CheckInvalidNew) {
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 1, 1, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 2, 1, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 3, 1, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 1, 0, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {2, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 2, 0, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {2, 3, 0};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 3, 0, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 0, 0, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(PRIMITIV_C_OK,
              ::primitivCreateShapeWithDims(dims, 8, 10, &shape));
  }
  {
    ::primitivShape_t* shape;
    uint32_t dims[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivCreateShapeWithDims(dims, 9, 10, &shape));
  }
}

TEST_F(CShapeTest, CheckNumElementsUnderRank) {
  ::primitivShape_t *src;
  uint32_t dims[] = {2, 3, 5, 7, 11, 13};
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivCreateShapeWithDims(dims, 6, 17, &src));
  uint32_t lower_volume;
  ::primitivGetShapeLowerVolume(src, 0, &lower_volume);
  EXPECT_EQ(1u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 1, &lower_volume);
  EXPECT_EQ(2u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 2, &lower_volume);
  EXPECT_EQ(2u * 3u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 3, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 4, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 5, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u, lower_volume);
  ::primitivGetShapeLowerVolume(src, 6, &lower_volume);
  EXPECT_EQ(2u * 3u * 5u * 7u * 11u * 13u, lower_volume);
  ::primitivDeleteShape(src);
}

TEST_F(CShapeTest, CheckString) {
  struct TestCase {
    vector<std::uint32_t> dims;
    std::uint32_t batch;
    std::string expected;
  };
  const vector<TestCase> test_cases {
    {{1}, 1, "[]x1"},
    {{1, 1}, 1, "[]x1"},
    {{}, 1, "[]x1"},
    {{1}, 1, "[]x1"},
    {{1, 1}, 1, "[]x1"},
    {{}, 2, "[]x2"},
    {{1}, 2, "[]x2"},
    {{1, 1}, 2, "[]x2"},
    {{2}, 1, "[2]x1"},
    {{2, 1}, 1, "[2]x1"},
    {{2, 3}, 1, "[2,3]x1"},
    {{2, 3, 1}, 1, "[2,3]x1"},
    {{2, 3, 5}, 1, "[2,3,5]x1"},
    {{2, 3, 5, 1}, 1, "[2,3,5]x1"},
    {{2}, 3, "[2]x3"},
    {{2, 1}, 3, "[2]x3"},
    {{2, 3}, 5, "[2,3]x5"},
    {{2, 3, 1}, 5, "[2,3]x5"},
    {{2, 3, 5}, 7, "[2,3,5]x7"},
    {{2, 3, 5, 1}, 7, "[2,3,5]x7"},
  };
  for (const TestCase &tc : test_cases) {
    ::primitivShape_t *shape;
    if (tc.dims.size() > 0) {
      ASSERT_EQ(PRIMITIV_C_OK,
                ::primitivCreateShapeWithDims(
                    &tc.dims[0], tc.dims.size(), tc.batch, &shape));
    } else {
      uint32_t dims[] = {};
      ASSERT_EQ(PRIMITIV_C_OK,
                ::primitivCreateShapeWithDims(dims, 0, tc.batch, &shape));
    }
    std::size_t length = 0u;
    ::primitivRepresentShapeAsString(shape, nullptr, &length);
    EXPECT_GT(length, 0u);
    char buffer[length];
    ::primitivRepresentShapeAsString(shape, buffer, &length);
    EXPECT_EQ(tc.expected, (std::string) buffer);
    ::primitivDeleteShape(shape);
  }
  {
    ::primitivShape_t* shape;
    ::primitivCreateShape(&shape);
    std::size_t length = 0u;
    ::primitivRepresentShapeAsString(shape, nullptr, &length);
    EXPECT_GT(length, 0u);
    char buffer[length];
    ::primitivRepresentShapeAsString(shape, buffer, &length);
    EXPECT_EQ("[]x1", (std::string) buffer);
    ::primitivDeleteShape(shape);
  }
}

TEST_F(CShapeTest, CheckCopy) {
  ::primitivShape_t *src1;
  ::primitivShape_t *src2;
  uint32_t dims1[] = {2, 3, 5};
  uint32_t dims2[] = {1, 4};
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivCreateShapeWithDims(dims1, 3, 7, &src1));
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivCreateShapeWithDims(dims2, 2, 9, &src2));
  PRIMITIV_C_BOOL valid;
  ::primitivIsShapeEqualTo(src1, src2, &valid);
  EXPECT_EQ(PRIMITIV_C_FALSE, valid);

  ::primitivShape_t *copied;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCloneShape(src1, &copied));
  ::primitivIsShapeEqualTo(src1, copied, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);
  ::primitivDeleteShape(copied);

  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCloneShape(src2, &copied));
  ::primitivIsShapeEqualTo(src2, copied, &valid);
  EXPECT_EQ(PRIMITIV_C_TRUE, valid);
  ::primitivDeleteShape(copied);

  ::primitivDeleteShape(src1);
  ::primitivDeleteShape(src2);
}

}  // namespace c
}  // namespace primitiv
