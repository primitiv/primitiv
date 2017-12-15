/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_CUDA_DEVICE_H_
#define PRIMITIV_C_CUDA_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

CAPI extern primitiv_Device *primitiv_CUDA_new(uint32_t device_id);
CAPI extern primitiv_Device *safe_primitiv_CUDA_new(uint32_t device_id,
                                                    primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_CUDA_new_with_seed(uint32_t device_id,
                                                         uint32_t rng_seed);
CAPI extern primitiv_Device *safe_primitiv_CUDA_new_with_seed(
    uint32_t device_id, uint32_t rng_seed, primitiv_Status *status);

CAPI extern void primitiv_CUDA_delete(primitiv_Device *device);
CAPI extern void safe_primitiv_CUDA_delete(primitiv_Device *device,
                                           primitiv_Status *status);

CAPI extern uint32_t primitiv_CUDA_num_devices();
CAPI extern uint32_t safe_primitiv_CUDA_num_devices(primitiv_Status *status);

CAPI extern void primitiv_CUDA_dump_description(const primitiv_Device *device);
CAPI extern void safe_primitiv_CUDA_dump_description(
    const primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
