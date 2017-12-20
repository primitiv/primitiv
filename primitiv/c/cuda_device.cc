/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/cuda_device.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/cuda_device.h>

using primitiv::devices::CUDA;

#define CAST_TO_CC_CUDA(x) reinterpret_cast<CUDA*>(x)
#define CAST_TO_CONST_CC_CUDA(x) reinterpret_cast<const CUDA*>(x)

extern "C" {

primitiv_Device *primitiv_CUDA_new(uint32_t device_id) {
  return to_c(new CUDA(device_id));
}
primitiv_Device *safe_primitiv_CUDA_new(uint32_t device_id,
                                        primitiv_Status *status) {
  SAFE_RETURN(primitiv_CUDA_new(device_id), status, nullptr);
}

primitiv_Device *primitiv_CUDA_new_with_seed(uint32_t device_id,
                                             uint32_t rng_seed) {
  return to_c(new CUDA(device_id, rng_seed));
}
primitiv_Device *safe_primitiv_CUDA_new_with_seed(uint32_t device_id,
                                                  uint32_t rng_seed,
                                                  primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_CUDA_new_with_seed(device_id, rng_seed), status, nullptr);
}

void primitiv_CUDA_delete(primitiv_Device *device) {
  delete CAST_TO_CC_CUDA(device);
}
void safe_primitiv_CUDA_delete(primitiv_Device *device,
                               primitiv_Status *status) {
  SAFE_EXPR(primitiv_CUDA_delete(device), status);
}

uint32_t primitiv_CUDA_num_devices() {
  return CUDA::num_devices();
}
uint32_t safe_primitiv_CUDA_num_devices(primitiv_Status *status) {
  SAFE_RETURN(primitiv_CUDA_num_devices(), status, 0);
}

void primitiv_CUDA_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_CUDA(device)->dump_description();
}
void safe_primitiv_CUDA_dump_description(const primitiv_Device *device,
                                         primitiv_Status *status) {
  SAFE_EXPR(primitiv_CUDA_dump_description(device), status);
}

}  // end extern "C"
