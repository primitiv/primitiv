#include <primitiv/config.h>

#include <primitiv/initializer.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/initializer.h>

using primitiv::Initializer;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitiv_Initializer_delete(
    primitiv_Initializer *initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  delete to_cpp_ptr(initializer);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Initializer_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_cpp_ptr(initializer)->apply(*to_cpp_ptr(x));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
