/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_OPENCL_DEVICE_H_
#define PRIMITIV_C_OPENCL_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

CAPI extern primitiv_Device *primitiv_OpenCL_new(uint32_t platform_id,
                                                 uint32_t device_id);
CAPI extern primitiv_Device *safe_primitiv_OpenCL_new(uint32_t platform_id,
                                                      uint32_t device_id,
                                                      primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_OpenCL_new_with_seed(uint32_t platform_id,
                                                           uint32_t device_id,
                                                           uint32_t rng_seed);
CAPI extern primitiv_Device *safe_primitiv_OpenCL_new_with_seed(
    uint32_t platform_id,
    uint32_t device_id,
    uint32_t rng_seed,
    primitiv_Status *status);

CAPI extern void primitiv_OpenCL_delete(primitiv_Device *device);
CAPI extern void safe_primitiv_OpenCL_delete(primitiv_Device *device,
                                             primitiv_Status *status);

CAPI extern uint32_t primitiv_OpenCL_num_platforms();
CAPI extern uint32_t safe_primitiv_OpenCL_num_platforms(
    primitiv_Status *status);

CAPI extern uint32_t primitiv_OpenCL_num_devices(uint32_t platform_id);
CAPI extern uint32_t safe_primitiv_OpenCL_num_devices(uint32_t platform_id,
                                                      primitiv_Status *status);

CAPI extern void primitiv_OpenCL_dump_description(
    const primitiv_Device *device);
CAPI extern void safe_primitiv_OpenCL_dump_description(
    const primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPENCL_DEVICE_H_
