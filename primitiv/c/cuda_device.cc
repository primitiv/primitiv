/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/cuda_device.h>

using primitiv::devices::CUDA;
using primitiv::c::internal::to_c;

#define CAST_TO_CC_CUDA(x) reinterpret_cast<CUDA*>(x)
#define CAST_TO_CONST_CC_CUDA(x) reinterpret_cast<const CUDA*>(x)

extern "C" {

primitiv_Status primitiv_CUDA_new(primitiv_Device **device,
                                  uint32_t device_id) {
  try {
    *device = to_c(new CUDA(device_id));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_CUDA_new_with_seed(
    primitiv_Device **device, uint32_t device_id, uint32_t rng_seed) {
  try {
    *device = to_c(new CUDA(device_id, rng_seed));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_CUDA_delete(primitiv_Device *device) {
  delete CAST_TO_CC_CUDA(device);
}

uint32_t primitiv_CUDA_num_devices() {
  return CUDA::num_devices();
}

void primitiv_CUDA_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_CUDA(device)->dump_description();
}

}  // end extern "C"

#undef CAST_TO_CC_CUDA
#undef CAST_TO_CONST_CC_CUDA
