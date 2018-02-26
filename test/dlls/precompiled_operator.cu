#include <cstdint>
#include <primitiv/error.h>
#include <primitiv/shape.h>
#include <primitiv/shape_ops.h>
#include <primitiv/tensor.h>

extern "C" {

void forward_shape(
    std::size_t argn, std::size_t retn,
    const primitiv::Shape *args, primitiv::Shape *rets) {
  if (argn != 2) PRIMITIV_THROW_ERROR("Invalid argument: argn != 2");
  if (retn != 1) PRIMITIV_THROW_ERROR("Invalid argument: retn != 1");
  rets[0] = primitiv::shape_ops::elementwise(args[0], args[1]);
}

void forward(
    std::size_t argn, std::size_t retn,
    const primitiv::Tensor *args, primitiv::Tensor *rets) {
  if (argn != 2) PRIMITIV_THROW_ERROR("Invalid argument: argn != 2");
  if (retn != 1) PRIMITIV_THROW_ERROR("Invalid argument: retn != 1");
  static_cast<void>(args);
  static_cast<void>(rets);
}

void backward(
    std::size_t argn, std::size_t retn,
    const primitiv::Tensor *arg_values, const primitiv::Tensor *ret_values,
    const primitiv::Tensor *ret_grads, primitiv::Tensor *arg_grads) {
  if (argn != 2) PRIMITIV_THROW_ERROR("Invalid argument: argn != 2");
  if (retn != 1) PRIMITIV_THROW_ERROR("Invalid argument: retn != 1");
  static_cast<void>(arg_values);
  static_cast<void>(ret_values);
  static_cast<void>(ret_grads);
  static_cast<void>(arg_grads);
}

}  // extern "C"
