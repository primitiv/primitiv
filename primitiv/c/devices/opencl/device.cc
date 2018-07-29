#include <primitiv/config.h>

#include <primitiv/c/devices/opencl/device.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/devices/opencl/device.h>

using primitiv::devices::OpenCL;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateOpenCLDevice(
    uint32_t platform_id, uint32_t device_id, primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new OpenCL(platform_id, device_id));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateOpenCLDeviceWithSeed(
    uint32_t platform_id, uint32_t device_id, uint32_t rng_seed,
    primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new OpenCL(platform_id, device_id, rng_seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNumOpenCLPlatforms(
    uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = OpenCL::num_platforms();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNumOpenCLDevices(
    uint32_t platform_id, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = OpenCL::num_devices(platform_id);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
