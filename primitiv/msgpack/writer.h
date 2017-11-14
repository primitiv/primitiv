#ifndef PRIMITIV_MSGPACK_WRITER_H_
#define PRIMITIV_MSGPACK_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <primitiv/error.h>
#include <primitiv/mixins.h>
#include <primitiv/msgpack/objects.h>

namespace primitiv {
namespace msgpack {

#define UC(expr) static_cast<char>(expr)

/**
 * ostream-like MessagePack writer.
 */
class Writer : mixins::Nonmovable<Writer> {
  std::ostream &os_;

private:
  Writer &write_string(const char *x, std::size_t size) {
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
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
    os_.write(x, size);
    return *this;
  }

public:
  /**
   * Creates a new Writer object.
   * @param os Target output stream.
   */
  Writer(std::ostream &os) : os_(os) {}

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
    static_assert(sizeof(float) == sizeof(std::uint32_t), "");
    std::uint32_t y;
    std::memcpy(&y, &x, sizeof(float));
    const char buf[5] {
      UC(0xca), UC(y >> 24), UC(y >> 16), UC(y >> 8), UC(y),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(double x) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "");
    std::uint64_t y;
    std::memcpy(&y, &x, sizeof(double));
    const char buf[9] {
      UC(0xcb),
      UC(y >> 56), UC(y >> 48), UC(y >> 40), UC(y >> 32),
      UC(y >> 24), UC(y >> 16), UC(y >> 8), UC(y),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(const char *x) {
    return write_string(x, std::strlen(x));
  }

  Writer &operator<<(const std::string &x) {
    return write_string(x.data(), x.size());
  }

  Writer &operator<<(const objects::Binary &x) {
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
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

  Writer &operator<<(const objects::Extension &x) {
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::int8_t type = x.type();
    const std::size_t size = x.size();
    if (size < (1ull << 8)) {
      switch (size) {
        case 1:
          {
            const char buf[2] { UC(0xd4), type };
            os_.write(buf, 2);
            break;
          }
        case 2:
          {
            const char buf[2] { UC(0xd5), type };
            os_.write(buf, 2);
            break;
          }
        case 4:
          {
            const char buf[2] { UC(0xd6), type };
            os_.write(buf, 2);
            break;
          }
        case 8:
          {
            const char buf[2] { UC(0xd7), type };
            os_.write(buf, 2);
            break;
          }
        case 16:
          {
            const char buf[2] { UC(0xd8), type };
            os_.write(buf, 2);
            break;
          }
        default:
          {
            const char buf[3] { UC(0xc7), UC(size), type };
            os_.write(buf, 3);
          }
      }
    } else if (size < (1ull << 16)) {
      const char buf[4] { UC(0xc8), UC(size >> 8), UC(size), type };
      os_.write(buf, 4);
    } else if (size < (1ull << 32)) {
      const char buf[6] {
        UC(0xc9),
        UC(size >> 24), UC(size >> 16), UC(size >> 8), UC(size),
        type,
      };
      os_.write(buf, 6);
    } else {
      THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one ext message.");
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
  }

  template<typename T>
  Writer &operator<<(const std::vector<T> &x) {
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
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
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
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

#endif  // PRIMITIV_MSGPACK_WRITER_H_
