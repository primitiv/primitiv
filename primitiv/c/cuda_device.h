/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_CUDA_DEVICE_H_
#define PRIMITIV_C_CUDA_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param device Pointer to receive a handler.
 * @return Status code.
 * @remarks The random number generator is initialized using
 *          `std::random_device`.
 */
CAPI extern primitiv_Status primitiv_devices_CUDA_new(
    uint32_t device_id, primitiv_Device **device);

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param rng_seed The seed value of the random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_devices_CUDA_new_with_seed(
    uint32_t device_id, uint32_t rng_seed, primitiv_Device **device);

/**
 * Deletes the Device object.
 * @param device Pointer of a handler.
 */
CAPI extern void primitiv_devices_CUDA_delete(primitiv_Device *device);

/**
 * Retrieves the number of active hardwares.
 * @return Number of active hardwares.
 */
CAPI extern uint32_t primitiv_devices_CUDA_num_devices();

/**
 * Prints device description to stderr.
 * @param device Pointer of a handler.
 */
CAPI extern void primitiv_devices_CUDA_dump_description(
    const primitiv_Device *device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
