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

#define DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(cc_name, c_name) \
inline c_name *to_c(primitiv::cc_name *instance) { \
  return reinterpret_cast<c_name*>(instance); \
} \
inline const c_name *to_c(const primitiv::cc_name *instance) { \
  return reinterpret_cast<const c_name*>(instance); \
} \
inline primitiv::cc_name *to_cc(c_name *instance) { \
  return reinterpret_cast<primitiv::cc_name*>(instance); \
} \
inline const primitiv::cc_name *to_cc(const c_name *instance) { \
  return reinterpret_cast<const primitiv::cc_name*>(instance); \
}

#define DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(cc_name, c_name) \
inline c_name *to_c_from_value(const primitiv::cc_name &instance) { \
  return reinterpret_cast<c_name*>(new primitiv::cc_name(instance)); \
}

#define HANDLE_EXCEPTION \
catch (const primitiv::Error &e) { \
  return primitiv_c::internal::ErrorHandler::get_instance().handle(e); \
} catch (const std::exception &e) { \
  return primitiv_c::internal::ErrorHandler::get_instance().handle(e); \
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

namespace primitiv_c {

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

DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Device, primitiv_Device);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Node, primitiv_Node);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Node, primitiv_Node);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Graph, primitiv_Graph);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Initializer, primitiv_Initializer);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Model, primitiv_Model);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Parameter, primitiv_Parameter);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Shape, primitiv_Shape);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Shape, primitiv_Shape);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Tensor, primitiv_Tensor);
DEFINE_VALUE_TO_POINTER_CONVERSION_AS_CAST(Tensor, primitiv_Tensor);
DEFINE_POINTER_TO_POINTER_CONVERSION_AS_CAST(Optimizer, primitiv_Optimizer);

}  // namespace internal

}  // namespace primitiv_c

#endif  // PRIMITIV_C_INTERNAL_H_
