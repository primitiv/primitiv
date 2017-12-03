#include "primitiv_c/cuda_device.h"

#include <primitiv/cuda_device.h>

using primitiv::devices::CUDA;

extern "C" {

primitiv_Device* primitiv_CUDA_new(uint32_t device_id) {
  return reinterpret_cast<primitiv_Device*>(new CUDA(device_id));
}

primitiv_Device* primitiv_CUDA_new_with_seed(uint32_t device_id, uint32_t rng_seed) {
  return reinterpret_cast<primitiv_Device*>(new CUDA(device_id, rng_seed));
}

void primitiv_CUDA_delete(primitiv_Device *device) {
  delete reinterpret_cast<CUDA*>(device);
}

uint32_t primitiv_CUDA_num_devices() {
  return CUDA::num_devices();
}

}  // end extern "C"
