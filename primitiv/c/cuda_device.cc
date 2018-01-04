#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/cuda_device.h>

using primitiv::devices::CUDA;
using primitiv::c::internal::to_c_ptr;

primitiv_Status primitiv_devices_CUDA_new(
    uint32_t device_id, primitiv_Device **device) try {
  *device = to_c_ptr(new CUDA(device_id));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_CUDA_new_with_seed(
    uint32_t device_id, uint32_t rng_seed, primitiv_Device **device) try {
  *device = to_c_ptr(new CUDA(device_id, rng_seed));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_CUDA_num_devices(uint32_t *num_devices) try {
  *num_devices = CUDA::num_devices();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
