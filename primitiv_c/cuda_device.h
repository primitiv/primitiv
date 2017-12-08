#ifndef PRIMITIV_C_CUDA_DEVICE_H_
#define PRIMITIV_C_CUDA_DEVICE_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Device* primitiv_CUDA_new(uint32_t device_id);
primitiv_Device* safe_primitiv_CUDA_new(uint32_t device_id, primitiv_Status *status);

primitiv_Device* primitiv_CUDA_new_with_seed(uint32_t device_id, uint32_t rng_seed);
primitiv_Device* safe_primitiv_CUDA_new_with_seed(uint32_t device_id, uint32_t rng_seed, primitiv_Status *status);

void primitiv_CUDA_delete(primitiv_Device *device);
void safe_primitiv_CUDA_delete(primitiv_Device *device, primitiv_Status *status);

uint32_t primitiv_CUDA_num_devices();
uint32_t safe_primitiv_CUDA_num_devices(primitiv_Status *status);

void primitiv_CUDA_dump_description(const primitiv_Device *device);
void safe_primitiv_CUDA_dump_description(const primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
