#ifndef PRIMITIV_C_INTERNAL_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_INTERNAL_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <primitiv/core/device.h>
#include <primitiv/core/error.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/initializer.h>
#include <primitiv/core/model.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>
#include <primitiv/core/optimizer.h>

#include <primitiv/c/define.h>

#define PRIMITIV_C_PTR_TO_PTR(cpp_name, c_name) \
inline c_name *to_c_ptr(primitiv::cpp_name *instance) { \
  return reinterpret_cast<c_name*>(instance); \
} \
inline const c_name *to_c_ptr(const primitiv::cpp_name *instance) { \
  return reinterpret_cast<const c_name*>(instance); \
} \
inline primitiv::cpp_name *to_cpp_ptr(c_name *instance) { \
  return reinterpret_cast<primitiv::cpp_name*>(instance); \
} \
inline const primitiv::cpp_name *to_cpp_ptr(const c_name *instance) { \
  return reinterpret_cast<const primitiv::cpp_name*>(instance); \
}

#define PRIMITIV_C_VAL_TO_PTR(cpp_name, c_name) \
inline c_name *to_c_ptr_from_value(primitiv::cpp_name &&instance) { \
  return reinterpret_cast<c_name*>( \
      new primitiv::cpp_name(std::forward<primitiv::cpp_name>(instance))); \
}

#define PRIMITIV_C_HANDLE_EXCEPTIONS \
catch (const std::exception &e) { \
  return primitiv::c::internal::ErrorHandler::get_instance().handle(e); \
}

#define PRIMITIV_C_CHECK_NOT_NULL(var) \
if (!var) { \
  PRIMITIV_THROW_ERROR("Argument `" #var "` must not be null."); \
}

struct primitivDevice;
struct primitivNode;
struct primitivGraph;
struct primitivInitializer;
struct primitivModel;
struct primitivParameter;
struct primitivShape;
struct primitivTensor;
struct primitivOptimizer;

namespace primitiv {

namespace c {

namespace internal {

template<typename T>
using Throwable = typename std::enable_if<
    std::is_base_of<std::exception, T>::value>::type;

class ErrorHandler {
 public:
  ErrorHandler() noexcept : exception_(nullptr), message_("OK") {}
  ~ErrorHandler() = default;

  template<typename T, typename = Throwable<T>>
  ::PRIMITIV_C_STATUS handle(const T &e) {
    exception_ = std::make_exception_ptr(e);
    message_ = e.what();
    return PRIMITIV_C_ERROR;
  }

  std::exception rethrow() {
    if (has_exception()) {
      std::rethrow_exception(exception_);
    } else {
      throw std::bad_exception();
    }
  }

  void reset() noexcept {
    exception_ = nullptr;
    message_ = "OK";
  }

  bool has_exception() const noexcept {
    return !exception_;
  }

  const char *get_message() const noexcept {
    return message_.c_str();
  }

  static ErrorHandler &get_instance();

 private:
  std::exception_ptr exception_;
  std::string message_;
};

PRIMITIV_C_PTR_TO_PTR(Device, primitivDevice);
PRIMITIV_C_PTR_TO_PTR(Node, primitivNode);
PRIMITIV_C_VAL_TO_PTR(Node, primitivNode);
PRIMITIV_C_PTR_TO_PTR(Graph, primitivGraph);
PRIMITIV_C_PTR_TO_PTR(Initializer, primitivInitializer);
PRIMITIV_C_PTR_TO_PTR(Model, primitivModel);
PRIMITIV_C_PTR_TO_PTR(Parameter, primitivParameter);
PRIMITIV_C_PTR_TO_PTR(Shape, primitivShape);
PRIMITIV_C_VAL_TO_PTR(Shape, primitivShape);
PRIMITIV_C_PTR_TO_PTR(Tensor, primitivTensor);
PRIMITIV_C_VAL_TO_PTR(Tensor, primitivTensor);
PRIMITIV_C_PTR_TO_PTR(Optimizer, primitivOptimizer);

template<typename T, typename U>
inline void move_vector_to_array_of_c_ptrs(
    std::vector<T> *src, U **array, std::size_t *size) {
  if (array) {
    if (*size < src->size()) {
      PRIMITIV_THROW_ERROR("Size is not enough to move a vector.");
    }
    std::transform(std::make_move_iterator(src->begin()),
                   std::make_move_iterator(src->end()),
                   array,
                   [](T &&x) {
                     return to_c_ptr_from_value(std::forward<T>(x));
                   });
  } else {
    *size = src->size();
  }
}

template<typename T>
inline void copy_vector_to_array(
    const std::vector<T> &src, T *array, std::size_t *size) {
  if (array) {
    if (*size < src.size()) {
      PRIMITIV_THROW_ERROR("Size is not enough to copy a vector.");
    }
    std::copy(src.begin(), src.end(), array);
  } else {
    *size = src.size();
  }
}

inline void copy_string_to_array(
    const std::string &str, char *buffer, std::size_t *size) {
  if (buffer) {
    if (*size <= str.length()) {
      PRIMITIV_THROW_ERROR("Size is not enough to copy a string.");
    }
    std::strcpy(buffer, str.c_str());
  } else {
    *size = str.length() + 1u;
  }
}

}  // namespace internal

}  // namespace c

}  // namespace primitiv

#endif  // PRIMITIV_C_INTERNAL_INTERNAL_H_
