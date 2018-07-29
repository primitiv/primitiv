#include <primitiv/config.h>

#include <vector>

#include <gtest/gtest.h>

#include <primitiv/c/internal/internal.h>

#include <test_utils.h>

using std::string;
using std::vector;
using test_utils::vector_match;

namespace primitiv {
namespace c {
namespace internal {

class CInternalTest : public testing::Test {};

TEST_F(CInternalTest, CheckCopyVectorToArray) {
  const vector<int> src1(1, 1), src2(2, 2), src3(3, 3);
  int dest[4] {};

  std::size_t size = 0u;
  int *dummy = nullptr;

  EXPECT_NO_THROW(copy_vector_to_array(src1, dummy, &size));
  EXPECT_EQ(1u, size);

  EXPECT_NO_THROW(copy_vector_to_array(src2, dummy, &size));
  EXPECT_EQ(2u, size);

  EXPECT_NO_THROW(copy_vector_to_array(src3, dummy, &size));
  EXPECT_EQ(3u, size);

  size = 2u;

  EXPECT_NO_THROW(copy_vector_to_array(src1, dest, &size));
  EXPECT_TRUE(
      vector_match(vector<int> {1, 0, 0, 0}, vector<int>(dest, dest + 4)));

  EXPECT_NO_THROW(copy_vector_to_array(src2, dest, &size));
  EXPECT_TRUE(
      vector_match(vector<int> {2, 2, 0, 0}, vector<int>(dest, dest + 4)));

  EXPECT_THROW(copy_vector_to_array(src3, dest, &size), Error);
  EXPECT_TRUE(
      vector_match(vector<int> {2, 2, 0, 0}, vector<int>(dest, dest + 4)));
}

TEST_F(CInternalTest, CheckCopyStringToArray) {
  const string src1 = "", src2 = "hello", src3 = "hello!";
  char dest[6] {};

  std::size_t size = 0u;
  char *dummy = nullptr;

  EXPECT_NO_THROW(copy_string_to_array(src1, dummy, &size));
  EXPECT_EQ(1u, size);

  EXPECT_NO_THROW(copy_string_to_array(src2, dummy, &size));
  EXPECT_EQ(6u, size);

  EXPECT_NO_THROW(copy_string_to_array(src3, dummy, &size));
  EXPECT_EQ(7u, size);

  size = 6u;

  EXPECT_NO_THROW(copy_string_to_array(src1, dest, &size));
  EXPECT_EQ("", string(dest));

  EXPECT_NO_THROW(copy_string_to_array(src2, dest, &size));
  EXPECT_EQ("hello", string(dest));

  EXPECT_THROW(copy_string_to_array(src3, dest, &size), Error);
  EXPECT_EQ("hello", string(dest));
}

}  // namespace internal
}  // namespace c
}  // namespace primitiv

