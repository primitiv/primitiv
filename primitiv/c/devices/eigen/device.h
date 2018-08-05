#ifndef PRIMITIV_C_DEVICES_EIGEN_DEVICE_H_
#define PRIMITIV_C_DEVICES_EIGEN_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 * @remarks The random number generator is initialized using
 *          `std::random_device`.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateEigenDevice(
    primitivDevice_t **newobj);

/**
 * Creates a new Device object.
 * @param rng_seed The seed value of the random number generator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateEigenDeviceWithSeed(
    uint32_t rng_seed, primitivDevice_t **newobj);

#endif  // PRIMITIV_C_DEVICES_EIGEN_DEVICE_H_
