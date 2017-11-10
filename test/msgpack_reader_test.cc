#include <config.h>

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
    ss = new std::istringstream(
        test_utils::bin_to_str(data) + static_cast<char>(0xc0));
    reader = new Reader(*ss);
  }

  void prepare_str(std::initializer_list<int> header, const string &data) {
    // Always adds 0xc0 (Nil) as the sentinel.
    ss = new std::istringstream(
        test_utils::bin_to_str(header) + data + static_cast<char>(0xc0));
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

  EXPECT_EQ(0x00000000, x[0]);
  EXPECT_EQ(0x7fffffff, x[1]);
  EXPECT_EQ(0x80000000, x[2]);
  EXPECT_EQ(0xffffffff, x[3]);
  EXPECT_EQ(0xdeadbeef, x[4]);
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

  EXPECT_EQ(0x00000000, x[0]);
  EXPECT_EQ(0x7fffffff, x[1]);
  EXPECT_EQ(0x80000000, x[2]);
  EXPECT_EQ(0xffffffff, x[3]);
  EXPECT_EQ(0xdeadbeef, x[4]);
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

  EXPECT_EQ(0x0000000000000000ll, x[0]);
  EXPECT_EQ(0x7fffffffffffffffll, x[1]);
  EXPECT_EQ(0x8000000000000000ll, x[2]);
  EXPECT_EQ(0xffffffffffffffffll, x[3]);
  EXPECT_EQ(0xdeadbeeffee1deadll, x[4]);
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

}  // namespace msgpack
}  // namespace primitiv
