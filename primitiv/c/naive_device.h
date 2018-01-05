#ifndef PRIMITIV_C_NAIVE_DEVICE_H_
#define PRIMITIV_C_NAIVE_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_devices_Naive_new(
    primitiv_Device **device);

/**
 * Creates a new Device object.
 * @param seed The seed value of internal random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_devices_Naive_new_with_seed(
    uint32_t seed, primitiv_Device **device);

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
