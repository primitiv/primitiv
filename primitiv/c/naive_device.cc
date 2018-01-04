#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitiv_devices_Naive_new(primitiv_Device **device) try {
  *device = to_c_ptr(new Naive());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_devices_Naive_new_with_seed(
    uint32_t seed, primitiv_Device **device) try {
  *device = to_c_ptr(new Naive(seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
