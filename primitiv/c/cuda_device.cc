#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/cuda_device.h>

using primitiv::devices::CUDA;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateCudaDevice(
    uint32_t device_id, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new CUDA(device_id));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateCudaDeviceWithSeed(
    uint32_t device_id, uint32_t rng_seed, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(new CUDA(device_id, rng_seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNumCudaDevices(uint32_t *num_devices) try {
  PRIMITIV_C_CHECK_NOT_NULL(num_devices);
  *num_devices = CUDA::num_devices();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
