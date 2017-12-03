#ifndef PRIMITIV_C_CUDA_DEVICE_H_
#define PRIMITIV_C_CUDA_DEVICE_H_

#include "primitiv_c/device.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Device* primitiv_CUDA_new(uint32_t device_id);

primitiv_Device* primitiv_CUDA_new_with_seed(uint32_t device_id, uint32_t rng_seed);

void primitiv_CUDA_delete(primitiv_Device *device);

uint32_t primitiv_CUDA_num_devices();

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
