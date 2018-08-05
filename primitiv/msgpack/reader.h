#ifndef PRIMITIV_MSGPACK_READER_H_
#define PRIMITIV_MSGPACK_READER_H_

#include <primitiv/config.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>
#include <primitiv/msgpack/objects.h>

namespace primitiv {
namespace msgpack {

/**
 * istream-like MessagePack reader.
 */
class Reader : mixins::Nonmovable<Reader> {
  std::istream &is_;

private:
  void check_eof() {
    if (!is_) {
      if (is_.eof()) {
        PRIMITIV_THROW_ERROR("MessagePack: Stream reached EOF.");
      } else {
        PRIMITIV_THROW_ERROR(
            "MessagePack: An error occurred while reading the stream.");
      }
    }
  }

  std::uint8_t get_uint8() {
    const std::uint8_t c = is_.get();
    check_eof();
    return c;
  }

  std::uint16_t get_uint16() {
    std::uint8_t c[2];
    is_.read(reinterpret_cast<char *>(c), 2);
    check_eof();
    return (c[0] << 8) | c[1];
  }

  std::uint32_t get_uint32() {
    std::uint8_t c[4];
    is_.read(reinterpret_cast<char *>(c), 4);
    check_eof();
    return (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];
  }

#define PRIMITIV_ULL(expr) static_cast<std::uint64_t>(expr)
  std::uint64_t get_uint64() {
    std::uint8_t c[8];
    is_.read(reinterpret_cast<char *>(c), 8);
    check_eof();
    return
      (PRIMITIV_ULL(c[0]) << 56) | (PRIMITIV_ULL(c[1]) << 48) |
      (PRIMITIV_ULL(c[2]) << 40) | (PRIMITIV_ULL(c[3]) << 32) |
      (PRIMITIV_ULL(c[4]) << 24) | (PRIMITIV_ULL(c[5]) << 16) |
      (PRIMITIV_ULL(c[6]) << 8) | PRIMITIV_ULL(c[7]);
  }
#undef PRIMITIV_ULL

  void read(char *ptr, std::size_t size) {
    is_.read(ptr, size);
    check_eof();
  }

  void check_type(std::uint8_t expected) {
    std::uint8_t observed = get_uint8();
    if (observed != expected) {
      PRIMITIV_THROW_ERROR(
          "MessagePack: Next object does not have a correct type. "
          "expected: " << std::hex << static_cast<int>(expected)
          << ", observed: " << std::hex << static_cast<int>(observed));
    }
  }

public:
  /**
   * Creates a new Reader object.
   * @param is Target input stream.
   */
  Reader(std::istream &is) : is_(is) {}

  Reader &operator>>(std::nullptr_t) {
    // Do nothing. Only checking the type.
    check_type(0xc0);
    return *this;
  }

  Reader &operator>>(bool &x) {
    const std::uint8_t type = get_uint8();
    if ((type & 0xfe) == 0xc2) {
      x = static_cast<bool>(type & 0x01);
    } else {
      PRIMITIV_THROW_ERROR(
          "MessagePack: Next object does not have the 'bool' type. "
          "observed: " << type);
    }
    return *this;
  }

  Reader &operator>>(std::uint8_t &x) {
    check_type(0xcc);
    x = get_uint8();
    return *this;
  }

  Reader &operator>>(std::uint16_t &x) {
    check_type(0xcd);
    x = get_uint16();
    return *this;
  }

  Reader &operator>>(std::uint32_t &x) {
    check_type(0xce);
    x = get_uint32();
    return *this;
  }

  Reader &operator>>(std::uint64_t &x) {
    check_type(0xcf);
    x = get_uint64();
    return *this;
  }

  Reader &operator>>(std::int8_t &x) {
    check_type(0xd0);
    x = get_uint8();
    return *this;
  }

  Reader &operator>>(std::int16_t &x) {
    check_type(0xd1);
    x = get_uint16();
    return *this;
  }

  Reader &operator>>(std::int32_t &x) {
    check_type(0xd2);
    x = get_uint32();
    return *this;
  }

  Reader &operator>>(std::int64_t &x) {
    check_type(0xd3);
    x = get_uint64();
    return *this;
  }

  Reader &operator>>(float &x) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "");
    check_type(0xca);
    const std::uint32_t y = get_uint32();
    std::memcpy(&x, &y, sizeof(float));
    return *this;
  }

  Reader &operator>>(double &x) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "");
    check_type(0xcb);
    const std::uint64_t y = get_uint64();
    std::memcpy(&x, &y, sizeof(double));
    return *this;
  }

  Reader &operator>>(std::string &x) {
    static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t), "");
    const std::uint8_t type = get_uint8();
    std::size_t size;
    if ((type & 0xe0) == 0xa0) {
      size = type & 0x1f;
    } else {
      switch (type) {
        case 0xd9: size = get_uint8(); break;
        case 0xda: size = get_uint16(); break;
        case 0xdb: size = get_uint32(); break;
        default:
          PRIMITIV_THROW_ERROR(
              "MessagePack: Next object does not have the 'str' type. "
              "observed: " << type);
      }
    }
    std::string ret(size, 0);
    read(&ret[0], size);
    x = std::move(ret);
    return *this;
  }

  Reader &operator>>(objects::Binary &x) {
    static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t), "");
    const std::uint8_t type = get_uint8();
    std::size_t size;
    switch (type) {
      case 0xc4: size = get_uint8(); break;
      case 0xc5: size = get_uint16(); break;
      case 0xc6: size = get_uint32(); break;
      default:
        PRIMITIV_THROW_ERROR(
            "MessagePack: Next object does not have the 'bin' type. "
            "observed: " << type);
    }
    objects::Binary ret;
    read(ret.allocate(size), size);
    x = std::move(ret);
    return *this;
  }

  Reader &operator>>(objects::Extension &x) {
    static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t), "");
    const std::uint8_t type = get_uint8();
    std::size_t size;
    switch (type) {
      case 0xd4: size = 1; break;
      case 0xd5: size = 2; break;
      case 0xd6: size = 4; break;
      case 0xd7: size = 8; break;
      case 0xd8: size = 16; break;
      case 0xc7: size = get_uint8(); break;
      case 0xc8: size = get_uint16(); break;
      case 0xc9: size = get_uint32(); break;
      default:
        PRIMITIV_THROW_ERROR(
            "MessagePack: Next object does not have the 'ext' type. "
            "observed: " << type);
    }
    objects::Extension ret;
    read(ret.allocate(get_uint8(), size), size);
    x = std::move(ret);
    return *this;
  }

  template<typename T>
  Reader &operator>>(std::vector<T> &x) {
    static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t), "");
    const std::uint8_t type = get_uint8();
    std::size_t size;
    if ((type & 0xf0) == 0x90) {
      size = type & 0x0f;
    } else {
      switch (type) {
        case 0xdc: size = get_uint16(); break;
        case 0xdd: size = get_uint32(); break;
        default:
          PRIMITIV_THROW_ERROR(
              "MessagePack: Next object does not have the 'array' type. "
              "observed: " << type);
      }
    }
    std::vector<T> ret(size);
    for (size_t i = 0; i < size; ++i) *this >> ret[i];
    x = std::move(ret);
    return *this;
  }

  template<typename T, typename U>
  Reader &operator>>(std::unordered_map<T, U> &x) {
    static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t), "");
    const std::uint8_t type = get_uint8();
    std::size_t size;
    if ((type & 0xf0) == 0x80) {
      size = type & 0x0f;
    } else {
      switch (type) {
        case 0xde: size = get_uint16(); break;
        case 0xdf: size = get_uint32(); break;
        default:
          PRIMITIV_THROW_ERROR(
              "MessagePack: Next object does not have the 'map' type. "
              "observed: " << type);
      }
    }
    std::unordered_map<T, U> ret;
    T key = T();
    U value = U();
    for (size_t i = 0; i < size; ++i) {
      *this >> key;
      *this >> value;
      ret.emplace(std::move(key), std::move(value));
    }
    x = std::move(ret);
    return *this;
  }
};

}  // namespace msgpack
}  // namespace primitiv

#endif  // PRIMITIV_MSGPACK_READER_H_
