/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_EIGEN_DEVICE_H_
#define PRIMITIV_C_EIGEN_DEVICE_H_

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
 * @remarks The random number generator is initialized using
 *          `std::random_device`.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_devices_Eigen_new(
    primitiv_Device **device);

/**
 * Creates a new Device object.
 * @param rng_seed The seed value of the random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_devices_Eigen_new_with_seed(
    uint32_t rng_seed, primitiv_Device **device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_EIGEN_DEVICE_H_
