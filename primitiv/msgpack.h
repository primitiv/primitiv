#ifndef PRIMITIV_MSGPACK_H_
#define PRIMITIV_MSGPACK_H_

// NOTE(odashi):
// This header implements the writer (emitter) and reader (parser) of
// the standard MessagePack wire format.
// Formal MessagePack specification can be found in:
//
//     https://github.com/msgpack/msgpack
//

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <primitiv/error.h>
#include <primitiv/mixins.h>

namespace primitiv {
namespace msgpack {

namespace writer_objects {

/**
 * Container to represent a binary object.
 */
class Binary : mixins::Nonmovable<Binary> {
  std::uint64_t size_;
  const void *data_;

public:
  /**
   * Creates a new Binary object.
   * @param size Number of bytes of the data.
   * @param data Pointer of raw data.
   */
  Binary(std::uint64_t size, const void *data) : size_(size), data_(data) {}

  /**
   * Retrieves the size of the data.
   * @return Size of the data.
   */
  std::uint64_t size() const { return size_; }

  /**
   * Retrieves the inner pointer.
   * @return Inner pointer.
   */
  const void *data() const { return data_; }
};

}  // namespace writer_objects

#define UC(expr) static_cast<char>(expr)

/**
 * IOStream-like MessagePack writer.
 */
class Writer : mixins::Nonmovable<Writer> {
  std::ostream &os_;

public:
  /**
   * Creates a new Writer object.
   * @param os Target output stream.
   */
  Writer(std::ostream &os) : os_(os) {};

  Writer &operator<<(std::nullptr_t x) {
    const char buf[1] { UC(0xc0) };
    os_.write(buf, 1);
    return *this;
  }

  Writer &operator<<(bool x) {
    const char buf[2] { UC(0xc2), UC(0xc3) };
    os_.write(&buf[!!x], 1);
    return *this;
  }

  Writer &operator<<(std::uint8_t x) {
    const char buf[2] { UC(0xcc), UC(x) };
    os_.write(buf, 2);
    return *this;
  }

  Writer &operator<<(std::uint16_t x) {
    const char buf[3] { UC(0xcd), UC(x >> 8), UC(x) };
    os_.write(buf, 3);
    return *this;
  }

  Writer &operator<<(std::uint32_t x) {
    const char buf[5] {
      UC(0xce), UC(x >> 24), UC(x >> 16), UC(x >> 8), UC(x),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(std::uint64_t x) {
    const char buf[9] {
      UC(0xcf),
      UC(x >> 56), UC(x >> 48), UC(x >> 40), UC(x >> 32),
      UC(x >> 24), UC(x >> 16), UC(x >> 8), UC(x),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(std::int8_t x) {
    const char buf[2] { UC(0xd0), UC(x) };
    os_.write(buf, 2);
    return *this;
  }

  Writer &operator<<(std::int16_t x) {
    const char buf[3] { UC(0xd1), UC(x >> 8), UC(x) };
    os_.write(buf, 3);
    return *this;
  }

  Writer &operator<<(std::int32_t x) {
    const char buf[5] {
      UC(0xd2), UC(x >> 24), UC(x >> 16), UC(x >> 8), UC(x),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(std::int64_t x) {
    const char buf[9] {
      UC(0xd3),
      UC(x >> 56), UC(x >> 48), UC(x >> 40), UC(x >> 32),
      UC(x >> 24), UC(x >> 16), UC(x >> 8), UC(x),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(float x) {
    union f2u32 {
      float f;
      std::uint32_t u;
    };
    const std::uint32_t y = reinterpret_cast<const f2u32 *>(&x)->u;
    const char buf[5] {
      UC(0xca), UC(y >> 24), UC(y >> 16), UC(y >> 8), UC(y),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(double x) {
    union d2u64 {
      double d;
      std::uint64_t u;
    };
    const std::uint64_t y = reinterpret_cast<const d2u64 *>(&x)->u;
    const char buf[9] {
      UC(0xcb),
      UC(y >> 56), UC(y >> 48), UC(y >> 40), UC(y >> 32),
      UC(y >> 24), UC(y >> 16), UC(y >> 8), UC(y),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(const char *x) {
    return operator<<(std::string(x));
  }

  Writer &operator<<(const std::string &x) {
    const std::uint64_t size = x.size();
    if (size < (1 << 5)) {
      const char buf[1] { UC(0xa0 | (size & 0x1f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 8)) {
      const char buf[2] { UC(0xd9), UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ull << 16)) {
      const char buf[3] { UC(0xda), UC(size >> 8), UC(size) };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        UC(0xdb), UC(size >> 24), UC(size >> 16), UC(size >> 8), UC(size),
      };
      os_.write(buf, 5);
    } else {
      THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one str message.");
    }
    os_.write(x.c_str(), size);
    return *this;
  }

  Writer &operator<<(const writer_objects::Binary &x) {
    const std::uint64_t size = x.size();
    if (size < (1ull << 8)) {
      const char buf[2] { UC(0xc4), UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ull << 16)) {
      const char buf[3] { UC(0xc5), UC(size >> 8), UC(size) };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        UC(0xc6), UC(size >> 24), UC(size >> 16), UC(size >> 8), UC(size),
      };
      os_.write(buf, 5);
    } else {
      THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one bin message.");
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
  }

  template<typename T>
  Writer &operator<<(const std::vector<T> &x) {
    const std::uint64_t size = x.size();
    if (size < (1ull << 4)) {
      const char buf[1] { UC(0x90 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 16)) {
      const char buf[3] { UC(0xdc), UC(size >> 8), UC(size) };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        UC(0xdd), UC(size >> 24), UC(size >> 16), UC(size >> 8), UC(size),
      };
      os_.write(buf, 5);
    }
    for (const T &elm : x) *this << elm;
    return *this;
  }

  template<typename T, typename U>
  Writer &operator<<(const std::unordered_map<T, U> &x) {
    const std::uint64_t size = x.size();
    if (size < (1ull << 4)) {
      const char buf[1] { UC(0x80 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 16)) {
      const char buf[3] { UC(0xde), UC(size >> 8), UC(size) };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        UC(0xdf), UC(size >> 24), UC(size >> 16), UC(size >> 8), UC(size),
      };
      os_.write(buf, 5);
    }
    for (const std::pair<T, U> &elm : x) *this << elm.first << elm.second;
    return *this;
  }
};

#undef UC

}  // namespace msgpack
}  // namespace primitiv

#endif  // PRIMITIV_MSGPACK_H_
