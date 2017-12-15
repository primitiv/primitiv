/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_NAIVE_DEVICE_H_
#define PRIMITIV_C_NAIVE_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

CAPI extern primitiv_Device *primitiv_Naive_new();
CAPI extern primitiv_Device *safe_primitiv_Naive_new(primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_Naive_new_with_seed(uint32_t seed);
CAPI extern primitiv_Device *safe_primitiv_Naive_new_with_seed(
    uint32_t seed, primitiv_Status *status);

CAPI extern void primitiv_Naive_delete(primitiv_Device *device);
CAPI extern void safe_primitiv_Naive_delete(primitiv_Device *device,
                                            primitiv_Status *status);

CAPI extern void primitiv_Naive_dump_description(const primitiv_Device *device);
CAPI extern void safe_primitiv_Naive_dump_description(
    const primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
