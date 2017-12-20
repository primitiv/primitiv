#include <primitiv/config.h>

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/string_utils.h>

using std::string;
using std::vector;

namespace primitiv {
namespace string_utils {

class StringUtilsTest : public testing::Test {};

TEST_F(StringUtilsTest, CheckJoin) {
  EXPECT_EQ("", join(vector<string> {}, ""));
  EXPECT_EQ("", join(vector<string> {}, "."));
  EXPECT_EQ("", join(vector<string> {}, "xxx"));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, ""));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, "."));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, "xxx"));
  EXPECT_EQ("foobar", join(vector<string> {"foo", "bar"}, ""));
  EXPECT_EQ("foo.bar", join(vector<string> {"foo", "bar"}, "."));
  EXPECT_EQ("fooxxxbar", join(vector<string> {"foo", "bar"}, "xxx"));
  EXPECT_EQ("foobarbaz", join(vector<string> {"foo", "bar", "baz"}, ""));
  EXPECT_EQ("foo.bar.baz", join(vector<string> {"foo", "bar", "baz"}, "."));
  EXPECT_EQ("fooxxxbarxxxbaz", join(vector<string> {"foo", "bar", "baz"}, "xxx"));
}

}  // namespace string_utils
}  // namespace primitiv
