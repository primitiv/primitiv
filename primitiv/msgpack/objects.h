#ifndef PRIMITIV_MSGPACK_OBJECTS_H_
#define PRIMITIV_MSGPACK_OBJECTS_H_

#include <primitiv/config.h>

#include <cstddef>
#include <cstdint>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/noncopyable.h>

namespace primitiv {
namespace msgpack {
namespace objects {

/**
 * Container to represent a binary object.
 */
class Binary : mixins::Noncopyable<Binary> {
  std::size_t size_;

  // NOTE(odashi):
  // Only one of eigher `ex_data_` or `in_data_` could be a valid pointer.
  const char *ex_data_;
  char *in_data_;

public:
  /**
   * Creates a placeholder object.
   */
  Binary() : size_(0), ex_data_(nullptr), in_data_(nullptr) {}

  /**
   * Move constructor.
   */
  Binary(Binary &&src)
    : size_(src.size_)
    , ex_data_(src.ex_data_)
    , in_data_(src.in_data_) {
      src.size_ = 0;
      src.ex_data_ = nullptr;
      src.in_data_ = nullptr;
    }

  /**
   * Move assignment.
   */
  Binary &operator=(Binary &&src) {
    if (&src != this) {
      delete in_data_;
      size_ = src.size_;
      ex_data_ = src.ex_data_;
      in_data_ = src.in_data_;
      src.size_ = 0;
      src.ex_data_ = nullptr;
      src.in_data_ = nullptr;
    }
    return *this;
  }

  /**
   * Creates a new `Binary` object with an external memory.
   * @param size Number of bytes of the data.
   * @param data Pointer of raw data.
   */
  Binary(std::size_t size, const char *data)
    : size_(size), ex_data_(data), in_data_(nullptr) {}

  ~Binary() {
    delete[] in_data_;
  }

  /**
   * Checks whether the object is valid or not.
   * @reutrn true the object is valid, false otherwise.
   */
  bool valid() const { return ex_data_ || in_data_; }

  /**
   * Returns whether the object is valid or not.
   * @throw primitiv::Error Object is invalid.
   */
  void check_valid() const {
    if (!valid()) PRIMITIV_THROW_ERROR("MessagePack: Invalid 'Binary' object.");
  }

  /**
   * Retrieves the size of the data.
   * @return Size of the data.
   */
  std::size_t size() const {
    check_valid();
    return size_;
  }

  /**
   * Retrieves the inner pointer.
   * @return Inner pointer.
   */
  const char *data() const {
    check_valid();
    return ex_data_ ? ex_data_ : in_data_;
  }

  /**
   * Initializes a new memory for the data and returns it.
   * @param size Number of bytes to be allocated for the data.
   * @remarks Allocated memory is managed by `Binary` object itself.
   *          Users must not delete memory returned by this function.
   */
  char *allocate(std::size_t size) {
    if (valid()) {
      PRIMITIV_THROW_ERROR("MessagePack: 'Binary' object is already valid.");
    }

    // NOTE(odashi):
    // Allocation should be done at first (it may throws).
    in_data_ = new char[size];

    size_ = size;
    return in_data_;
  }
};

/**
 * Container to represent an extension object.
 */
class Extension : mixins::Noncopyable<Extension> {
  std::int8_t type_;
  std::size_t size_;

  // NOTE(odashi):
  // Only one of eigher `ex_data_` or `in_data_` could be a valid pointer.
  const char *ex_data_;
  char *in_data_;

public:
  /**
   * Creates a placeholder object.
   */
  Extension() : type_(0), size_(0), ex_data_(nullptr), in_data_(nullptr) {}

  /**
   * Move constructor.
   */
  Extension(Extension &&src)
    : type_(src.type_)
    , size_(src.size_)
    , ex_data_(src.ex_data_)
    , in_data_(src.in_data_) {
      src.type_ = 0;
      src.size_ = 0;
      src.ex_data_ = nullptr;
      src.in_data_ = nullptr;
    }

  /**
   * Move assignment.
   */
  Extension &operator=(Extension &&src) {
    if (&src != this) {
      delete in_data_;
      type_ = src.type_;
      size_ = src.size_;
      ex_data_ = src.ex_data_;
      in_data_ = src.in_data_;
      src.type_ = 0;
      src.size_ = 0;
      src.ex_data_ = nullptr;
      src.in_data_ = nullptr;
    }
    return *this;
  }

  /**
   * Creates a new `Extension` object with an external memory.
   * @param type Extension type.
   * @param size Number of bytes of the data.
   * @param data Pointer of raw data.
   */
  Extension(std::int8_t type, std::size_t size, const char *data)
    : type_(type), size_(size), ex_data_(data), in_data_(nullptr) {}

  ~Extension() {
    delete[] in_data_;
  }

  /**
   * Returns whether the object is valid or not.
   * @reutrn true the object is valid, false otherwise.
   */
  bool valid() const { return ex_data_ || in_data_; }

  /**
   * Returns whether the object is valid or not.
   * @throw primitiv::Error Object is invalid.
   */
  void check_valid() const {
    if (!valid()) {
      PRIMITIV_THROW_ERROR("MessagePack: Invalid 'Extension' object.");
    }
  }

  /**
   * Retrieves the type of this extension.
   * @return Type of this extension.
   */
  std::int8_t type() const {
    check_valid();
    return type_;
  }

  /**
   * Retrieves the size of the data.
   * @return Size of the data.
   */
  std::size_t size() const {
    check_valid();
    return size_;
  }

  /**
   * Retrieves the inner pointer.
   * @return Inner pointer.
   */
  const char *data() const {
    check_valid();
    return ex_data_ ? ex_data_ : in_data_;
  }

  /**
   * Initializes a new memory for the data and returns it.
   * @param type Extension type.
   * @param size Number of bytes to be allocated for the data.
   * @remarks Allocated memory is managed by `Extension` object itself.
   *          Users must not delete memory returned by this function.
   */
  char *allocate(std::int8_t type, std::size_t size) {
    if (valid()) {
      PRIMITIV_THROW_ERROR("MessagePack: 'Extension' object is already valid.");
  }

    // NOTE(odashi):
    // Allocation should be done at first (it may throws).
    in_data_ = new char[size];

    type_ = type;
    size_ = size;
    return in_data_;
  }
};

}  // namespace objects
}  // namespace msgpack
}  // namespace primitiv

#endif  // PRIMITIV_MSGPACK_OBJECTS_H_
