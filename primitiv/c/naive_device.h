/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_NAIVE_DEVICE_H_
#define PRIMITIV_C_NAIVE_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new Device object.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_devices_Naive_new(
    primitiv_Device **device);

/**
 * Creates a new Device object.
 * @param seed The seed value of internal random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_devices_Naive_new_with_seed(
    uint32_t seed, primitiv_Device **device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
