#include <config.h>

#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/msgpack.h>
#include <test_utils.h>

using std::string;
using std::unordered_map;
using std::vector;
using test_utils::bin_to_str;

namespace primitiv {
namespace msgpack {

class WriterTest : public testing::Test {
protected:
  std::stringstream ss;
  Writer writer;

  void match(std::initializer_list<int> data) {
    EXPECT_EQ(bin_to_str(data), ss.str());
  }

  void match_str(std::initializer_list<int> header, const std::string &data) {
    EXPECT_EQ(bin_to_str(header) + data, ss.str());
  }

  void match_header_and_size(std::initializer_list<int> header, unsigned size) {
    EXPECT_EQ(bin_to_str(header), ss.str().substr(0, header.size()));
    EXPECT_EQ(size, ss.str().size());
  }

public:
  WriterTest() : ss(), writer(ss) {}
};

TEST_F(WriterTest, CheckNil) {
  writer << nullptr;
  match({ 0xc0 });
}

TEST_F(WriterTest, CheckBoolFalse) {
  writer << false;
  match({ 0xc2 });
}

TEST_F(WriterTest, CheckBoolTrue) {
  writer << true;
  match({ 0xc3 });
}

TEST_F(WriterTest, CheckUInt8) {
  writer << static_cast<std::uint8_t>(0x12);
  match({ 0xcc, 0x12 });
}

TEST_F(WriterTest, CheckUInt16) {
  writer << static_cast<std::uint16_t>(0x1234);
  match({ 0xcd, 0x12, 0x34 });
}

TEST_F(WriterTest, CheckUInt32) {
  writer << static_cast<std::uint32_t>(0x12345678);
  match({ 0xce, 0x12, 0x34, 0x56, 0x78 });
}

TEST_F(WriterTest, CheckUInt64) {
  writer << static_cast<std::uint64_t>(0x123456789abcdef0ull);
  match({ 0xcf, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0 });
}

TEST_F(WriterTest, CheckInt8) {
  writer << static_cast<std::int8_t>(0x12);
  match({ 0xd0, 0x12 });
}

TEST_F(WriterTest, CheckInt16) {
  writer << static_cast<std::int16_t>(0x1234);
  match({ 0xd1, 0x12, 0x34 });
}

TEST_F(WriterTest, CheckInt32) {
  writer << static_cast<std::int32_t>(0x12345678);
  match({ 0xd2, 0x12, 0x34, 0x56, 0x78 });
}

TEST_F(WriterTest, CheckInt64) {
  writer << static_cast<std::int64_t>(0x123456789abcdef0ull);
  match({ 0xd3, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0 });
}

TEST_F(WriterTest, CheckFloat) {
  writer << 1.f;
  match({ 0xca, 0x3f, 0x80, 0x00, 0x00 });
}

TEST_F(WriterTest, CheckDouble) {
  writer << 1.;
  match({ 0xcb, 0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 });
}

TEST_F(WriterTest, CheckString_0) {
  writer << "";
  match({ 0xa0 });
}

TEST_F(WriterTest, CheckString_1) {
  writer << "a";
  match_str({ 0xa1 }, "a");
}

TEST_F(WriterTest, CheckString_31) {
  const string &data = "1234567890123456789012345678901";  // 31 chars
  writer << data;
  match_str({ 0xbf }, data);
}

TEST_F(WriterTest, CheckString_32) {
  const string &data = "12345678901234567890123456789012";  // 32 chars
  writer << data;
  match_str({ 0xd9, 0x20 }, data);
}

TEST_F(WriterTest, CheckString_0xff) {
  vector<char> buf(0xff, 'a');
  const string data(buf.begin(), buf.end());
  writer << data;
  match_str({ 0xd9, 0xff }, data);
}

TEST_F(WriterTest, CheckString_0x100) {
  vector<char> buf(0x100, 'a');
  const string data(buf.begin(), buf.end());
  writer << data;
  match_str({ 0xda, 0x01, 0x00 }, data);
}

TEST_F(WriterTest, CheckString_0xffff) {
  vector<char> buf(0xffff, 'a');
  const string data(buf.begin(), buf.end());
  writer << data;
  match_str({ 0xda, 0xff, 0xff }, data);
}

TEST_F(WriterTest, CheckString_0x10000) {
  vector<char> buf(0x10000, 'a');
  const string data(buf.begin(), buf.end());
  writer << data;
  match_str({ 0xdb, 0x00, 0x01, 0x00, 0x00 }, data);
}

TEST_F(WriterTest, CheckBinary_0) {
  writer << writer_objects::Binary("", 0);
  match({ 0xc4, 0x00 });
}

TEST_F(WriterTest, CheckBinary_1) {
  writer << writer_objects::Binary("a", 1);
  match_str({ 0xc4, 0x01 }, "a");
}

TEST_F(WriterTest, CheckBinary_0xff) {
  vector<char> buf(0xff, 'a');
  const string data(buf.begin(), buf.end());
  writer << writer_objects::Binary(data.c_str(), 0xff);
  match_str({ 0xc4, 0xff }, data);
}

TEST_F(WriterTest, CheckBinary_0x100) {
  vector<char> buf(0x100, 'a');
  const string data(buf.begin(), buf.end());
  writer << writer_objects::Binary(data.c_str(), 0x100);
  match_str({ 0xc5, 0x01, 0x00 }, data);
}

TEST_F(WriterTest, CheckBinary_0xffff) {
  vector<char> buf(0xffff, 'a');
  const string data(buf.begin(), buf.end());
  writer << writer_objects::Binary(data.c_str(), 0xffff);
  match_str({ 0xc5, 0xff, 0xff }, data);
}

TEST_F(WriterTest, CheckBinary_0x10000) {
  vector<char> buf(0x10000, 'a');
  const string data(buf.begin(), buf.end());
  writer << writer_objects::Binary(data.c_str(), 0x10000);
  match_str({ 0xc6, 0x00, 0x01, 0x00, 0x00 }, data);
}

TEST_F(WriterTest, CheckVector_Nil_0) {
  vector<std::nullptr_t> vec;
  writer << vec;
  match({ 0x90 });
}

TEST_F(WriterTest, CheckVector_Nil_1) {
  vector<std::nullptr_t> vec(1);
  writer << vec;
  match({ 0x91, 0xc0 });
}

TEST_F(WriterTest, CheckVector_Nil_15) {
  vector<std::nullptr_t> vec(15);
  vector<char> buf(15, 0xc0);
  writer << vec;
  match_str({ 0x9f }, string(buf.begin(), buf.end()));
}

TEST_F(WriterTest, CheckVector_Nil_16) {
  vector<std::nullptr_t> vec(16);
  vector<char> buf(16, 0xc0);
  writer << vec;
  match_str({ 0xdc, 0x00, 0x10 }, string(buf.begin(), buf.end()));
}

TEST_F(WriterTest, CheckVector_Nil_0xffff) {
  vector<std::nullptr_t> vec(0xffff);
  vector<char> buf(0xffff, 0xc0);
  writer << vec;
  match_str({ 0xdc, 0xff, 0xff }, string(buf.begin(), buf.end()));
}

TEST_F(WriterTest, CheckVector_Nil_0x10000) {
  vector<std::nullptr_t> vec(0x10000);
  vector<char> buf(0x10000, 0xc0);
  writer << vec;
  match_str({ 0xdd, 0x00, 0x01, 0x00, 0x00 }, string(buf.begin(), buf.end()));
}

TEST_F(WriterTest, CheckVector_UInt8_3) {
  vector<std::uint8_t> vec { 0x11, 0x22, 0x33 };
  writer << vec;
  match({ 0x93, 0xcc, 0x11, 0xcc, 0x22, 0xcc, 0x33 });
}

TEST_F(WriterTest, CheckVector_String_2) {
  vector<string> vec { "foo", "bar" };
  writer << vec;
  match({ 0x92, 0xa3, 'f', 'o', 'o', 0xa3, 'b', 'a', 'r' });
}

TEST_F(WriterTest, CheckMap_Bool_Nil_0) {
  unordered_map<bool, nullptr_t> mp;
  writer << mp;
  match({ 0x80 });
}

TEST_F(WriterTest, CheckMap_Bool_Nil_1) {
  unordered_map<bool, nullptr_t> mp;
  mp.emplace(false, nullptr);
  writer << mp;
  match({ 0x81, 0xc2, 0xc0 });
}

TEST_F(WriterTest, CheckMap_UInt8_Nil_15) {
  unordered_map<std::int8_t, nullptr_t> mp;
  for (int8_t i = 0; i < 15; ++i) mp.emplace(i, nullptr);
  writer << mp;

  // Can't expect a correct data order.
  match_header_and_size({ 0x8f }, 1 + 3 * 15);
}

TEST_F(WriterTest, CheckMap_UInt8_Nil_16) {
  unordered_map<std::int8_t, nullptr_t> mp;
  for (int8_t i = 0; i < 16; ++i) mp.emplace(i, nullptr);
  writer << mp;

  // Can't expect a correct data order.
  match_header_and_size({ 0xde, 0x00, 0x10 }, 3 + 3 * 16);
}

TEST_F(WriterTest, CheckMap_UInt32_Nil_0xffff) {
  unordered_map<std::int32_t, nullptr_t> mp;
  for (int32_t i = 0; i < 0xffff; ++i) mp.emplace(i, nullptr);
  writer << mp;

  // Can't expect a correct data order.
  match_header_and_size({ 0xde, 0xff, 0xff }, 3 + 6 * 0xffff);
}

TEST_F(WriterTest, CheckMap_UInt8_Nil_0x10000) {
  unordered_map<std::int32_t, nullptr_t> mp;
  for (int32_t i = 0; i < 0x10000; ++i) mp.emplace(i, nullptr);
  writer << mp;

  // Can't expect a correct data order.
  match_header_and_size({ 0xdf, 0x00, 0x01, 0x00, 0x00 }, 5 + 6 * 0x10000);
}

TEST_F(WriterTest, CheckUserSequence) {
  writer << nullptr << true << static_cast<std::uint16_t>(0x1234) << 1.f;
  match({
      0xc0,  // nullptr
      0xc3,  // true
      0xcd, 0x12, 0x34,  // uint16_t
      0xca, 0x3f, 0x80, 0x00, 0x00,  // float
  });
}

}  // namespace msgpack
}  // namespace primitiv
