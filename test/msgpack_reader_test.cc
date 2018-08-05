#include <primitiv/config.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/msgpack/reader.h>

#include <test_utils.h>

using std::string;
using std::unordered_map;
using std::vector;
using test_utils::bin_to_str;
using test_utils::vector_match;

namespace primitiv {
namespace msgpack {

class ReaderTest : public testing::Test {
protected:
  std::istringstream *ss;
  Reader *reader;

  void SetUp() override {
    ss = nullptr;
    reader = nullptr;
  }

  void TearDown() override {
    delete reader;
    delete ss;
  }

  void prepare(std::initializer_list<int> data) {
    // Always adds 0xc0 (Nil) as the sentinel.
    ss = new std::istringstream(bin_to_str(data) + static_cast<char>(0xc0));
    reader = new Reader(*ss);
  }

  void prepare_str(std::initializer_list<int> header, const string &data) {
    // Always adds 0xc0 (Nil) as the sentinel.
    ss = new std::istringstream(
        bin_to_str(header) + data + static_cast<char>(0xc0));
    reader = new Reader(*ss);
  }
};

TEST_F(ReaderTest, CheckEOF) {
  prepare({});
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_THROW(*reader >> nullptr, Error);  // Exceeds EOF
}

TEST_F(ReaderTest, CheckNil) {
  prepare({ 0xc0, 0xc0 });
  EXPECT_NO_THROW(*reader >> nullptr);
  EXPECT_NO_THROW(*reader >> nullptr);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
}

TEST_F(ReaderTest, CheckBool) {
  prepare({ 0xc2, 0xc3, 0xc2, 0xc3 });
  bool x[4] { true, false, true, false };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_FALSE(x[0]);
  EXPECT_TRUE(x[1]);
  EXPECT_FALSE(x[2]);
  EXPECT_TRUE(x[3]);
}

TEST_F(ReaderTest, CheckUInt8) {
  prepare({ 0xcc, 0x00, 0xcc, 0x7f, 0xcc, 0x80, 0xcc, 0xff, 0xcc, 0x42 });
  std::uint8_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x00, x[0]);
  EXPECT_EQ(0x7f, x[1]);
  EXPECT_EQ(0x80, x[2]);
  EXPECT_EQ(0xff, x[3]);
  EXPECT_EQ(0x42, x[4]);
}

TEST_F(ReaderTest, CheckUInt16) {
  prepare({
      0xcd, 0x00, 0x00, 0xcd, 0x7f, 0xff, 0xcd, 0x80, 0x00,
      0xcd, 0xff, 0xff, 0xcd, 0x12, 0x34,
  });
  std::uint16_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x0000, x[0]);
  EXPECT_EQ(0x7fff, x[1]);
  EXPECT_EQ(0x8000, x[2]);
  EXPECT_EQ(0xffff, x[3]);
  EXPECT_EQ(0x1234, x[4]);
}

TEST_F(ReaderTest, CheckUInt32) {
  prepare({
      0xce, 0x00, 0x00, 0x00, 0x00, 0xce, 0x7f, 0xff, 0xff, 0xff,
      0xce, 0x80, 0x00, 0x00, 0x00, 0xce, 0xff, 0xff, 0xff, 0xff,
      0xce, 0xde, 0xad, 0xbe, 0xef,
  });
  std::uint32_t x[5] {1, 2, 3, 4, 5};
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x00000000u, x[0]);
  EXPECT_EQ(0x7fffffffu, x[1]);
  EXPECT_EQ(0x80000000u, x[2]);
  EXPECT_EQ(0xffffffffu, x[3]);
  EXPECT_EQ(0xdeadbeefu, x[4]);
}

TEST_F(ReaderTest, CheckUInt64) {
  prepare({
      0xcf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xcf, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xcf, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xcf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xcf, 0xde, 0xad, 0xbe, 0xef, 0xfe, 0xe1, 0xde, 0xad,
  });
  std::uint64_t x[5] {1, 2, 3, 4, 5};
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x0000000000000000ull, x[0]);
  EXPECT_EQ(0x7fffffffffffffffull, x[1]);
  EXPECT_EQ(0x8000000000000000ull, x[2]);
  EXPECT_EQ(0xffffffffffffffffull, x[3]);
  EXPECT_EQ(0xdeadbeeffee1deadull, x[4]);
}

TEST_F(ReaderTest, CheckInt8) {
  prepare({ 0xd0, 0x00, 0xd0, 0x7f, 0xd0, 0x80, 0xd0, 0xff, 0xd0, 0x42 });
  std::int8_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(static_cast<std::int8_t>(0x00), x[0]);
  EXPECT_EQ(static_cast<std::int8_t>(0x7f), x[1]);
  EXPECT_EQ(static_cast<std::int8_t>(0x80), x[2]);
  EXPECT_EQ(static_cast<std::int8_t>(0xff), x[3]);
  EXPECT_EQ(static_cast<std::int8_t>(0x42), x[4]);
}

TEST_F(ReaderTest, CheckInt16) {
  prepare({
      0xd1, 0x00, 0x00, 0xd1, 0x7f, 0xff, 0xd1, 0x80, 0x00,
      0xd1, 0xff, 0xff, 0xd1, 0x12, 0x34,
  });
  std::int16_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(static_cast<std::int16_t>(0x0000), x[0]);
  EXPECT_EQ(static_cast<std::int16_t>(0x7fff), x[1]);
  EXPECT_EQ(static_cast<std::int16_t>(0x8000), x[2]);
  EXPECT_EQ(static_cast<std::int16_t>(0xffff), x[3]);
  EXPECT_EQ(static_cast<std::int16_t>(0x1234), x[4]);
}

TEST_F(ReaderTest, CheckInt32) {
  prepare({
      0xd2, 0x00, 0x00, 0x00, 0x00, 0xd2, 0x7f, 0xff, 0xff, 0xff,
      0xd2, 0x80, 0x00, 0x00, 0x00, 0xd2, 0xff, 0xff, 0xff, 0xff,
      0xd2, 0xde, 0xad, 0xbe, 0xef,
  });
  std::int32_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x00000000u, static_cast<std::uint32_t>(x[0]));
  EXPECT_EQ(0x7fffffffu, static_cast<std::uint32_t>(x[1]));
  EXPECT_EQ(0x80000000u, static_cast<std::uint32_t>(x[2]));
  EXPECT_EQ(0xffffffffu, static_cast<std::uint32_t>(x[3]));
  EXPECT_EQ(0xdeadbeefu, static_cast<std::uint32_t>(x[4]));
}

TEST_F(ReaderTest, CheckInt64) {
  prepare({
      0xd3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xd3, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xd3, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xd3, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xd3, 0xde, 0xad, 0xbe, 0xef, 0xfe, 0xe1, 0xde, 0xad,
  });
  std::int64_t x[5] { 1, 2, 3, 4, 5 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2] >> x[3] >> x[4]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  EXPECT_EQ(0x0000000000000000ull, static_cast<std::uint64_t>(x[0]));
  EXPECT_EQ(0x7fffffffffffffffull, static_cast<std::uint64_t>(x[1]));
  EXPECT_EQ(0x8000000000000000ull, static_cast<std::uint64_t>(x[2]));
  EXPECT_EQ(0xffffffffffffffffull, static_cast<std::uint64_t>(x[3]));
  EXPECT_EQ(0xdeadbeeffee1deadull, static_cast<std::uint64_t>(x[4]));
}

TEST_F(ReaderTest, CheckFloat) {
  prepare({
      0xca, 0x00, 0x00, 0x00, 0x00,
      0xca, 0x3f, 0x80, 0x00, 0x00,
      0xca, 0xc0, 0x40, 0x00, 0x00,
  });
  float x[3] { 1e10f, 1e10f, 1e10f };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  // x[i] should be completely equal to the left-hand side.
  EXPECT_EQ(0.f, x[0]);
  EXPECT_EQ(1.f, x[1]);
  EXPECT_EQ(-3.f, x[2]);
}

TEST_F(ReaderTest, CheckDouble) {
  prepare({
      0xcb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xcb, 0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xcb, 0xc0, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  });
  double x[3] { 1e10, 1e10, 1e10 };
  EXPECT_NO_THROW(*reader >> x[0] >> x[1] >> x[2]);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  // x[i] should be completely equal to the left-hand side.
  EXPECT_EQ(0., x[0]);
  EXPECT_EQ(1., x[1]);
  EXPECT_EQ(-3., x[2]);
}

TEST_F(ReaderTest, CheckString_0) {
  prepare({ 0xa0 });
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ("", x);
}

TEST_F(ReaderTest, CheckString_1) {
  prepare({ 0xa1, 'x' });
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ("x", x);
}

TEST_F(ReaderTest, CheckString_31) {
  const string data = "1234567890123456789012345678901";  // 31 chars
  prepare_str({ 0xbf }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckString_32) {
  const string data = "12345678901234567890123456789012";  // 32 chars
  prepare_str({ 0xd9, 0x20 }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckString_0xff) {
  const string data(0xff, 'a');
  prepare_str({ 0xd9, 0xff }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckString_0x100) {
  const string data(0x100, 'b');
  prepare_str({ 0xda, 0x01, 0x00 }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckString_0xffff) {
  const string data(0xffff, 'c');
  prepare_str({ 0xda, 0xff, 0xff }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckString_0x10000) {
  const string data(0x10000, 'd');
  prepare_str({ 0xdb, 0x00, 0x01, 0x00, 0x00 }, data);
  string x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(data, x);
}

TEST_F(ReaderTest, CheckBinary_0) {
  prepare({ 0xc4, 0x00 });
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ(0u, x.size());
}

TEST_F(ReaderTest, CheckBinary_1) {
  const string expected = "x";
  prepare({ 0xc4, 0x01, 'x' });
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(1u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckBinary_0xff) {
  const string expected(0xff, 'a');
  prepare_str({ 0xc4, 0xff }, expected);
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0xffu, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckBinary_0x100) {
  const string expected(0x100, 'b');
  prepare_str({ 0xc5, 0x01, 0x00 }, expected);
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0x100u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckBinary_0xffff) {
  const string expected(0xffff, 'c');
  prepare_str({ 0xc5, 0xff, 0xff }, expected);
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0xffffu, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckBinary_0x10000) {
  const string expected(0x10000, 'd');
  prepare_str({ 0xc6, 0x00, 0x01, 0x00, 0x00 }, expected);
  objects::Binary x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0x10000u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_0) {
  prepare({ 0xc7, 0x00, 'X' });
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  EXPECT_EQ(0u, x.size());
}

TEST_F(ReaderTest, CheckExtension_1) {
  const string expected = "1";
  prepare_str({ 0xd4, 'X' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(1u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_2) {
  const string expected = "12";
  prepare_str({ 0xd5, 'X' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(2u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_4) {
  const string expected = "1234";
  prepare_str({ 0xd6, 'X' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(4u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_8) {
  const string expected = "12345678";
  prepare_str({ 0xd7, 'X' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(8u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_16) {
  const string expected = "1234567890123456";
  prepare_str({ 0xd8, 'X' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('X', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(16u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_0xff) {
  const string expected(0xff, 'a');
  prepare_str({ 0xc7, 0xff, 'A' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('A', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0xffu, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_0x100) {
  const string expected(0x100, 'b');
  prepare_str({ 0xc8, 0x01, 0x00, 'B' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('B', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0x100u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_0xffff) {
  const string expected(0xffff, 'c');
  prepare_str({ 0xc8, 0xff, 0xff, 'C' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('C', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0xffffu, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckExtension_0x10000) {
  const string expected(0x10000, 'd');
  prepare_str({ 0xc9, 0x00, 0x01, 0x00, 0x00, 'D' }, expected);
  objects::Extension x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel

  ASSERT_NO_THROW(x.check_valid());
  EXPECT_EQ('D', x.type());
  const std::size_t size = x.size();
  const char *data = x.data();
  EXPECT_EQ(0x10000u, size);
  EXPECT_EQ(expected, string(data, size));
}

TEST_F(ReaderTest, CheckVector_Nil_0) {
  prepare({ 0x90 });
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_TRUE(x.empty());
}

TEST_F(ReaderTest, CheckVector_Nil_1) {
  prepare({ 0x91, 0xc0 });
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(1u, x.size());
  for (const std::nullptr_t y : x) EXPECT_EQ(nullptr, y);
}

TEST_F(ReaderTest, CheckVector_Nil_15) {
  prepare_str({ 0x9f }, string(15, 0xc0));
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(15u, x.size());
  for (const std::nullptr_t y : x) EXPECT_EQ(nullptr, y);
}

TEST_F(ReaderTest, CheckVector_Nil_16) {
  prepare_str({ 0xdc, 0x00, 0x10 }, string(16, 0xc0));
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(16u, x.size());
  for (const std::nullptr_t y : x) EXPECT_EQ(nullptr, y);
}

TEST_F(ReaderTest, CheckVector_Nil_0xffff) {
  prepare_str({ 0xdc, 0xff, 0xff }, string(0xffff, 0xc0));
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(0xffffu, x.size());
  for (const std::nullptr_t y : x) EXPECT_EQ(nullptr, y);
}

TEST_F(ReaderTest, CheckVector_Nil_0x10000) {
  prepare_str({ 0xdd, 0x00, 0x01, 0x00, 0x00 }, string(0x10000, 0xc0));
  vector<std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(0x10000u, x.size());
  for (const std::nullptr_t y : x) EXPECT_EQ(nullptr, y);
}

TEST_F(ReaderTest, CheckVector_UInt8_3) {
  prepare({ 0x93, 0xcc, 0x11, 0xcc, 0x22, 0xcc, 0x33 });
  vector<std::uint8_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_TRUE(vector_match(vector<std::uint8_t> { 0x11, 0x22, 0x33 }, x));
}

TEST_F(ReaderTest, CheckVector_String_2) {
  prepare({ 0x92, 0xa3, 'f', 'o', 'o', 0xa3, 'b', 'a', 'r' });
  vector<string> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_TRUE(vector_match(vector<string> { "foo", "bar" }, x));
}

TEST_F(ReaderTest, CheckMap_Bool_Nil_0) {
  prepare({ 0x80 });
  unordered_map<bool, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_TRUE(x.empty());
}

TEST_F(ReaderTest, CheckMap_Bool_Nil_1) {
  prepare({ 0x81, 0xc2, 0xc0 });
  unordered_map<bool, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(1u, x.size());
  EXPECT_EQ(nullptr, x.at(false));
}

TEST_F(ReaderTest, CheckMap_UInt8_Nil_15) {
  std::ostringstream ss;
  for (int i = 0; i < 15; ++i) ss << bin_to_str({ 0xcc, i, 0xc0 });
  prepare_str({ 0x8f }, ss.str());
  unordered_map<std::uint8_t, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(15u, x.size());
  for (int i = 0; i < 15; ++i) EXPECT_EQ(nullptr, x.at(i));
}

TEST_F(ReaderTest, CheckMap_UInt8_Nil_16) {
  std::ostringstream ss;
  for (int i = 0; i < 16; ++i) ss << bin_to_str({ 0xcc, i, 0xc0 });
  prepare_str({ 0xde, 0x00, 0x10 }, ss.str());
  unordered_map<std::uint8_t, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(16u, x.size());
  for (int i = 0; i < 15; ++i) EXPECT_EQ(nullptr, x.at(i));
}

TEST_F(ReaderTest, CheckMap_UInt32_Nil_0xffff) {
  std::ostringstream ss;
  for (int i = 0; i < 0xffff; ++i) {
    ss << bin_to_str({
        0xce, i >> 24, (i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff, 0xc0,
    });
  }
  prepare_str({ 0xde, 0xff, 0xff }, ss.str());
  unordered_map<std::uint32_t, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(0xffffu, x.size());
  for (int i = 0; i < 0xffff; ++i) EXPECT_EQ(nullptr, x.at(i));
}

TEST_F(ReaderTest, CheckMap_UInt32_Nil_0x10000) {
  std::ostringstream ss;
  for (int i = 0; i < 0x10000; ++i) {
    ss << bin_to_str({
        0xce, i >> 24, (i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff, 0xc0,
    });
  }
  prepare_str({ 0xdf, 0x00, 0x01, 0x00, 0x00 }, ss.str());
  unordered_map<std::uint32_t, std::nullptr_t> x;
  EXPECT_NO_THROW(*reader >> x);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(0x10000u, x.size());
  for (int i = 0; i < 0x10000; ++i) EXPECT_EQ(nullptr, x.at(i));
}

TEST_F(ReaderTest, CheckUserSequence) {
  prepare({
      0xc0,  // nullptr
      0xc3,  // true
      0xcd, 0x12, 0x34,  // uint16_t
      0xca, 0x3f, 0x80, 0x00, 0x00,  // float
  });
  std::nullptr_t x1 = nullptr;
  bool x2 = false;
  std::uint16_t x3 = 0;
  float x4 = 0.f;
  EXPECT_NO_THROW(*reader >> x1 >> x2 >> x3 >> x4);
  EXPECT_NO_THROW(*reader >> nullptr);  // Sentinel
  EXPECT_EQ(nullptr, x1);
  EXPECT_EQ(true, x2);
  EXPECT_EQ(0x1234, x3);
  EXPECT_EQ(1.f, x4);
}

}  // namespace msgpack
}  // namespace primitiv
