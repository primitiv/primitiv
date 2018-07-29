#include <primitiv/config.h>

#include <primitiv/core/device.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/device.h>

using primitiv::Device;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitivGetDefaultDevice(primitivDevice_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&Device::get_default());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetDefaultDevice(primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  Device::set_default(*to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteDevice(primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  delete to_cpp_ptr(device);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDumpDeviceDescription(
    const primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  to_cpp_ptr(device)->dump_description();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
