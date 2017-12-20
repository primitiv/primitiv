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
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Device *primitiv_devices_Naive_new();

/**
 * Creates a new Device object.
 * @param seed The seed value of internal random number generator.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Device *primitiv_devices_Naive_new_with_seed(
    uint32_t seed);

/**
 * Deletes the Device object.
 * @param device Pointer of a handler.
 */
CAPI extern void primitiv_devices_Naive_delete(primitiv_Device *device);

/**
 * Prints device description to stderr.
 * @param device Pointer of a handler.
 */
CAPI extern void primitiv_devices_Naive_dump_description(
    const primitiv_Device *device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
