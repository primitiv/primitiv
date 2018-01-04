#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

primitiv_Status primitiv_devices_Naive_new(primitiv_Device **device) try {
  *device = to_c_ptr(new Naive());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_Naive_new_with_seed(
    uint32_t seed, primitiv_Device **device) try {
  *device = to_c_ptr(new Naive(seed));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
