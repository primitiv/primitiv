#include <primitiv/config.h>

#include <utility>

#include <gtest/gtest.h>

#include <primitiv/msgpack/objects.h>

namespace primitiv {
namespace msgpack {
namespace objects {

class BinaryTest : public testing::Test {};

TEST_F(BinaryTest, CheckInvalid) {
  Binary obj;
  EXPECT_FALSE(obj.valid());
  EXPECT_THROW(obj.check_valid(), Error);
  EXPECT_THROW(obj.size(), Error);
  EXPECT_THROW(obj.data(), Error);
}

TEST_F(BinaryTest, CheckExternalData) {
  char data[] = "1234567890";
  Binary obj(10, data);
  EXPECT_TRUE(obj.valid());
  EXPECT_NO_THROW(obj.check_valid());
  EXPECT_EQ(10u, obj.size());
  EXPECT_EQ(data, obj.data());
  EXPECT_THROW(obj.allocate(42), Error);
}

TEST_F(BinaryTest, CheckInternalData) {
  Binary obj;
  EXPECT_NO_THROW(obj.allocate(42));
  EXPECT_TRUE(obj.valid());
  EXPECT_NO_THROW(obj.check_valid());
  EXPECT_EQ(42u, obj.size());
  EXPECT_NE(nullptr, obj.data());
  EXPECT_THROW(obj.allocate(123), Error);
}

TEST_F(BinaryTest, CheckMoveCtorWithExternalData) {
  char data[] = "1234567890";
  Binary org(10, data);

  Binary moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ(10u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(BinaryTest, CheckMoveAssignWithExternalData) {
  char data[] = "1234567890";
  Binary org(10, data);

  Binary moved;
  moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ(10u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(BinaryTest, CheckMoveCtorWithInternalData) {
  Binary org;
  const char *data = org.allocate(42);

  Binary moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ(42u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(BinaryTest, CheckMoveAssignWithInternalData) {
  Binary org;
  const char *data = org.allocate(42);

  Binary moved;
  moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ(42u, moved.size());
  EXPECT_EQ(data, moved.data());
}

class ExtensionTest : public testing::Test {};

TEST_F(ExtensionTest, CheckInvalid) {
  Extension obj;
  EXPECT_FALSE(obj.valid());
  EXPECT_THROW(obj.check_valid(), Error);
  EXPECT_THROW(obj.type(), Error);
  EXPECT_THROW(obj.size(), Error);
  EXPECT_THROW(obj.data(), Error);
}

TEST_F(ExtensionTest, CheckExternalData) {
  char data[] = "1234567890";
  Extension obj('X', 10, data);
  EXPECT_TRUE(obj.valid());
  EXPECT_NO_THROW(obj.check_valid());
  EXPECT_EQ('X', obj.type());
  EXPECT_EQ(10u, obj.size());
  EXPECT_EQ(data, obj.data());
  EXPECT_THROW(obj.allocate('Y', 42), Error);
}

TEST_F(ExtensionTest, CheckInternalData) {
  Extension obj;
  EXPECT_NO_THROW(obj.allocate('Y', 42));
  EXPECT_TRUE(obj.valid());
  EXPECT_NO_THROW(obj.check_valid());
  EXPECT_EQ('Y', obj.type());
  EXPECT_EQ(42u, obj.size());
  EXPECT_NE(nullptr, obj.data());
  EXPECT_THROW(obj.allocate('Z', 123), Error);
}

TEST_F(ExtensionTest, CheckMoveCtorWithExternalData) {
  char data[] = "1234567890";
  Extension org('X', 10, data);

  Extension moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ('X', moved.type());
  EXPECT_EQ(10u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(ExtensionTest, CheckMoveAssignWithExternalData) {
  char data[] = "1234567890";
  Extension org('X', 10, data);

  Extension moved;
  moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ('X', moved.type());
  EXPECT_EQ(10u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(ExtensionTest, CheckMoveCtorWithInternalData) {
  Extension org;
  const char *data = org.allocate('X', 42);

  Extension moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ('X', moved.type());
  EXPECT_EQ(42u, moved.size());
  EXPECT_EQ(data, moved.data());
}

TEST_F(ExtensionTest, CheckMoveAssignWithInternalData) {
  Extension org;
  const char *data = org.allocate('X', 42);

  Extension moved;
  moved = std::move(org);
  ASSERT_FALSE(org.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ('X', moved.type());
  EXPECT_EQ(42u, moved.size());
  EXPECT_EQ(data, moved.data());
}

}  // namespace objects
}  // namespace msgpack
}  // namespace primitiv
