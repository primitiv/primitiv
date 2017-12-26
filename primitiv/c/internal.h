/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_INTERNAL_H_
#define PRIMITIV_C_INTERNAL_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>
#include <primitiv/initializer.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>
#include <primitiv/optimizer.h>
#include <primitiv/c/status.h>

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
inline c_name *to_c_ptr_from_value(const primitiv::cpp_name &instance) { \
  return reinterpret_cast<c_name*>(new primitiv::cpp_name(instance)); \
}

#define PRIMITIV_C_HANDLE_EXCEPTIONS \
catch (const primitiv::Error &e) { \
  return primitiv::c::internal::ErrorHandler::get_instance().handle(e); \
} catch (const std::exception &e) { \
  return primitiv::c::internal::ErrorHandler::get_instance().handle(e); \
}

struct primitiv_Device;
struct primitiv_Node;
struct primitiv_Graph;
struct primitiv_Initializer;
struct primitiv_Model;
struct primitiv_Parameter;
struct primitiv_Shape;
struct primitiv_Tensor;
struct primitiv_Optimizer;

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
  ::primitiv_Status handle(const T &e) {
    exception_ = std::make_exception_ptr(e);
    message_ = e.what();
    return ::primitiv_Status::PRIMITIV_ERROR;
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

PRIMITIV_C_PTR_TO_PTR(Device, primitiv_Device);
PRIMITIV_C_PTR_TO_PTR(Node, primitiv_Node);
PRIMITIV_C_VAL_TO_PTR(Node, primitiv_Node);
PRIMITIV_C_PTR_TO_PTR(Graph, primitiv_Graph);
PRIMITIV_C_PTR_TO_PTR(Initializer, primitiv_Initializer);
PRIMITIV_C_PTR_TO_PTR(Model, primitiv_Model);
PRIMITIV_C_PTR_TO_PTR(Parameter, primitiv_Parameter);
PRIMITIV_C_PTR_TO_PTR(Shape, primitiv_Shape);
PRIMITIV_C_VAL_TO_PTR(Shape, primitiv_Shape);
PRIMITIV_C_PTR_TO_PTR(Tensor, primitiv_Tensor);
PRIMITIV_C_VAL_TO_PTR(Tensor, primitiv_Tensor);
PRIMITIV_C_PTR_TO_PTR(Optimizer, primitiv_Optimizer);

}  // namespace internal

}  // namespace c

}  // namespace primitiv

#endif  // PRIMITIV_C_INTERNAL_H_
