#include <primitiv/config.h>

#include <primitiv/c/devices/naive/device.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/devices/naive/device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateNaiveDevice(primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Naive());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateNaiveDeviceWithSeed(
    uint32_t seed, primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Naive(seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
