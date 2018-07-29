#include <primitiv/config.h>

#include <primitiv/c/devices/cuda/device.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/devices/cuda/device.h>

using primitiv::devices::CUDA;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateCudaDevice(
    uint32_t device_id, primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new CUDA(device_id));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateCudaDeviceWithSeed(
    uint32_t device_id, uint32_t rng_seed, primitivDevice_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new CUDA(device_id, rng_seed));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetNumCudaDevices(uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = CUDA::num_devices();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
