#include <primitiv/config.h>

#include <primitiv/core/initializer.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/initializer.h>

using primitiv::Initializer;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitivDeleteInitializer(
    primitivInitializer_t *initializer) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  delete to_cpp_ptr(initializer);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyInitializer(
    const primitivInitializer_t *initializer, primitivTensor_t *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_cpp_ptr(initializer)->apply(*to_cpp_ptr(x));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
