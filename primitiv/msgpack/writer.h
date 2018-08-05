#ifndef PRIMITIV_MSGPACK_WRITER_H_
#define PRIMITIV_MSGPACK_WRITER_H_

#include <primitiv/config.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>
#include <primitiv/msgpack/objects.h>

namespace primitiv {
namespace msgpack {

#define PRIMITIV_UC(expr) static_cast<char>(expr)

/**
 * ostream-like MessagePack writer.
 */
class Writer : mixins::Nonmovable<Writer> {
  std::ostream &os_;

private:
  Writer &write_string(const char *x, std::size_t size) {
#ifdef PRIMITIV_WORDSIZE_64
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    if (size < (1 << 5)) {
      const char buf[1] { PRIMITIV_UC(0xa0 | (size & 0x1f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 8)) {
      const char buf[2] { PRIMITIV_UC(0xd9), PRIMITIV_UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ull << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xda), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        PRIMITIV_UC(0xdb),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    } else {
      PRIMITIV_THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one str message.");
    }
    os_.write(x, size);
    return *this;
#else
    static_assert(sizeof(std::size_t) == sizeof(std::uint32_t), "");
    if (size < (1 << 5)) {
      const char buf[1] { PRIMITIV_UC(0xa0 | (size & 0x1f)) };
      os_.write(buf, 1);
    } else if (size < (1ul << 8)) {
      const char buf[2] { PRIMITIV_UC(0xd9), PRIMITIV_UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ul << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xda), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else {
      const char buf[5] {
        PRIMITIV_UC(0xdb),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    os_.write(x, size);
    return *this;
#endif
  }

public:
  /**
   * Creates a new Writer object.
   * @param os Target output stream.
   */
  Writer(std::ostream &os) : os_(os) {}

  Writer &operator<<(std::nullptr_t) {
    const char buf[1] { PRIMITIV_UC(0xc0) };
    os_.write(buf, 1);
    return *this;
  }

  Writer &operator<<(bool x) {
    const char buf[2] { PRIMITIV_UC(0xc2), PRIMITIV_UC(0xc3) };
    os_.write(&buf[!!x], 1);
    return *this;
  }

  Writer &operator<<(std::uint8_t x) {
    const char buf[2] { PRIMITIV_UC(0xcc), PRIMITIV_UC(x) };
    os_.write(buf, 2);
    return *this;
  }

  Writer &operator<<(std::uint16_t x) {
    const char buf[3] {
      PRIMITIV_UC(0xcd), PRIMITIV_UC(x >> 8), PRIMITIV_UC(x)
    };
    os_.write(buf, 3);
    return *this;
  }

  Writer &operator<<(std::uint32_t x) {
    const char buf[5] {
      PRIMITIV_UC(0xce),
      PRIMITIV_UC(x >> 24), PRIMITIV_UC(x >> 16),
      PRIMITIV_UC(x >> 8), PRIMITIV_UC(x),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(std::uint64_t x) {
    const char buf[9] {
      PRIMITIV_UC(0xcf),
      PRIMITIV_UC(x >> 56), PRIMITIV_UC(x >> 48),
      PRIMITIV_UC(x >> 40), PRIMITIV_UC(x >> 32),
      PRIMITIV_UC(x >> 24), PRIMITIV_UC(x >> 16),
      PRIMITIV_UC(x >> 8), PRIMITIV_UC(x),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(std::int8_t x) {
    const char buf[2] { PRIMITIV_UC(0xd0), PRIMITIV_UC(x) };
    os_.write(buf, 2);
    return *this;
  }

  Writer &operator<<(std::int16_t x) {
    const char buf[3] {
      PRIMITIV_UC(0xd1), PRIMITIV_UC(x >> 8), PRIMITIV_UC(x)
    };
    os_.write(buf, 3);
    return *this;
  }

  Writer &operator<<(std::int32_t x) {
    const char buf[5] {
      PRIMITIV_UC(0xd2),
      PRIMITIV_UC(x >> 24), PRIMITIV_UC(x >> 16),
      PRIMITIV_UC(x >> 8), PRIMITIV_UC(x),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(std::int64_t x) {
    const char buf[9] {
      PRIMITIV_UC(0xd3),
      PRIMITIV_UC(x >> 56), PRIMITIV_UC(x >> 48),
      PRIMITIV_UC(x >> 40), PRIMITIV_UC(x >> 32),
      PRIMITIV_UC(x >> 24), PRIMITIV_UC(x >> 16),
      PRIMITIV_UC(x >> 8), PRIMITIV_UC(x),
    };
    os_.write(buf, 9);
    return *this;
  }

  Writer &operator<<(float x) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "");
    std::uint32_t y;
    std::memcpy(&y, &x, sizeof(float));
    const char buf[5] {
      PRIMITIV_UC(0xca),
      PRIMITIV_UC(y >> 24), PRIMITIV_UC(y >> 16),
      PRIMITIV_UC(y >> 8), PRIMITIV_UC(y),
    };
    os_.write(buf, 5);
    return *this;
  }

  Writer &operator<<(double x) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "");
    std::uint64_t y;
    std::memcpy(&y, &x, sizeof(double));
    const char buf[9] {
      PRIMITIV_UC(0xcb),
      PRIMITIV_UC(y >> 56), PRIMITIV_UC(y >> 48),
      PRIMITIV_UC(y >> 40), PRIMITIV_UC(y >> 32),
      PRIMITIV_UC(y >> 24), PRIMITIV_UC(y >> 16),
      PRIMITIV_UC(y >> 8), PRIMITIV_UC(y),
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
#ifdef PRIMITIV_WORDSIZE_64
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ull << 8)) {
      const char buf[2] { PRIMITIV_UC(0xc4), PRIMITIV_UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ull << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xc5), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        PRIMITIV_UC(0xc6),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    } else {
      PRIMITIV_THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one bin message.");
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
#else
    static_assert(sizeof(std::size_t) == sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ul << 8)) {
      const char buf[2] { PRIMITIV_UC(0xc4), PRIMITIV_UC(size) };
      os_.write(buf, 2);
    } else if (size < (1ul << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xc5), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else {
      const char buf[5] {
        PRIMITIV_UC(0xc6),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
#endif
  }

  Writer &operator<<(const objects::Extension &x) {
#ifdef PRIMITIV_WORDSIZE_64
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::int8_t type = x.type();
    const std::size_t size = x.size();
    if (size < (1ull << 8)) {
      switch (size) {
        case 1:
          {
            const char buf[2] { PRIMITIV_UC(0xd4), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 2:
          {
            const char buf[2] { PRIMITIV_UC(0xd5), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 4:
          {
            const char buf[2] { PRIMITIV_UC(0xd6), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 8:
          {
            const char buf[2] { PRIMITIV_UC(0xd7), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 16:
          {
            const char buf[2] { PRIMITIV_UC(0xd8), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        default:
          {
            const char buf[3] {
              PRIMITIV_UC(0xc7), PRIMITIV_UC(size), PRIMITIV_UC(type)
            };
            os_.write(buf, 3);
          }
      }
    } else if (size < (1ull << 16)) {
      const char buf[4] {
        PRIMITIV_UC(0xc8), PRIMITIV_UC(size >> 8),
        PRIMITIV_UC(size), PRIMITIV_UC(type)
      };
      os_.write(buf, 4);
    } else if (size < (1ull << 32)) {
      const char buf[6] {
        PRIMITIV_UC(0xc9),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
        PRIMITIV_UC(type),
      };
      os_.write(buf, 6);
    } else {
      PRIMITIV_THROW_ERROR(
          "MessagePack: Can't store more than 2^32 - 1 bytes "
          "in one ext message.");
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
#else
    static_assert(sizeof(std::size_t) == sizeof(std::uint32_t), "");
    const std::int8_t type = x.type();
    const std::size_t size = x.size();
    if (size < (1ul << 8)) {
      switch (size) {
        case 1:
          {
            const char buf[2] { PRIMITIV_UC(0xd4), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 2:
          {
            const char buf[2] { PRIMITIV_UC(0xd5), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 4:
          {
            const char buf[2] { PRIMITIV_UC(0xd6), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 8:
          {
            const char buf[2] { PRIMITIV_UC(0xd7), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        case 16:
          {
            const char buf[2] { PRIMITIV_UC(0xd8), PRIMITIV_UC(type) };
            os_.write(buf, 2);
            break;
          }
        default:
          {
            const char buf[3] {
              PRIMITIV_UC(0xc7), PRIMITIV_UC(size), PRIMITIV_UC(type)
            };
            os_.write(buf, 3);
          }
      }
    } else if (size < (1ul << 16)) {
      const char buf[4] {
        PRIMITIV_UC(0xc8), PRIMITIV_UC(size >> 8),
        PRIMITIV_UC(size), PRIMITIV_UC(type)
      };
      os_.write(buf, 4);
    } else {
      const char buf[6] {
        PRIMITIV_UC(0xc9),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
        PRIMITIV_UC(type),
      };
      os_.write(buf, 6);
    }
    os_.write(reinterpret_cast<const char *>(x.data()), size);
    return *this;
#endif
  }

  template<typename T>
  Writer &operator<<(const std::vector<T> &x) {
#ifdef PRIMITIV_WORDSIZE_64
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ull << 4)) {
      const char buf[1] { PRIMITIV_UC(0x90 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xdc), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        PRIMITIV_UC(0xdd),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    for (const T &elm : x) *this << elm;
    return *this;
#else
    static_assert(sizeof(std::size_t) == sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ul << 4)) {
      const char buf[1] { PRIMITIV_UC(0x90 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ul << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xdc), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else {
      const char buf[5] {
        PRIMITIV_UC(0xdd),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    for (const T &elm : x) *this << elm;
    return *this;
#endif
  }

  template<typename T, typename U>
  Writer &operator<<(const std::unordered_map<T, U> &x) {
#ifdef PRIMITIV_WORDSIZE_64
    static_assert(sizeof(std::size_t) > sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ull << 4)) {
      const char buf[1] { PRIMITIV_UC(0x80 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ull << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xde), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else if (size < (1ull << 32)) {
      const char buf[5] {
        PRIMITIV_UC(0xdf),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    for (const std::pair<T, U> &elm : x) *this << elm.first << elm.second;
    return *this;
#else
    static_assert(sizeof(std::size_t) == sizeof(std::uint32_t), "");
    const std::size_t size = x.size();
    if (size < (1ul << 4)) {
      const char buf[1] { PRIMITIV_UC(0x80 | (size & 0x0f)) };
      os_.write(buf, 1);
    } else if (size < (1ul << 16)) {
      const char buf[3] {
        PRIMITIV_UC(0xde), PRIMITIV_UC(size >> 8), PRIMITIV_UC(size)
      };
      os_.write(buf, 3);
    } else {
      const char buf[5] {
        PRIMITIV_UC(0xdf),
        PRIMITIV_UC(size >> 24), PRIMITIV_UC(size >> 16),
        PRIMITIV_UC(size >> 8), PRIMITIV_UC(size),
      };
      os_.write(buf, 5);
    }
    for (const std::pair<T, U> &elm : x) *this << elm.first << elm.second;
    return *this;
#endif
  }
};

#undef PRIMITIV_UC

}  // namespace msgpack
}  // namespace primitiv

#endif  // PRIMITIV_MSGPACK_WRITER_H_
