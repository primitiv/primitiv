#include <primitiv/config.h>

#include <primitiv/opencl_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/opencl_device.h>

using primitiv::devices::OpenCL;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitiv_devices_OpenCL_new(
    uint32_t platform_id, uint32_t device_id, primitiv_Device **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new OpenCL(platform_id, device_id));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_devices_OpenCL_new_with_seed(
    uint32_t platform_id, uint32_t device_id, uint32_t rng_seed,
    primitiv_Device **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new OpenCL(platform_id, device_id, rng_seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_devices_OpenCL_num_platforms(
    uint32_t *num_platforms) try {
  PRIMITIV_C_CHECK_NOT_NULL(num_platforms);
  *num_platforms = OpenCL::num_platforms();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_devices_OpenCL_num_devices(
    uint32_t platform_id, uint32_t *num_devices) try {
  PRIMITIV_C_CHECK_NOT_NULL(num_devices);
  *num_devices = OpenCL::num_devices(platform_id);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
