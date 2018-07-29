#ifndef PRIMITIV_C_DEVICES_NAIVE_DEVICE_H_
#define PRIMITIV_C_DEVICES_NAIVE_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateNaiveDevice(
    primitivDevice_t **newobj);

/**
 * Creates a new Device object.
 * @param seed The seed value of internal random number generator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateNaiveDeviceWithSeed(
    uint32_t seed, primitivDevice_t **newobj);

#endif  // PRIMITIV_C_DEVICES_NAIVE_DEVICE_H_
