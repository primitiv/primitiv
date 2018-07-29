#ifndef PRIMITIV_C_DEVICES_OPENCL_DEVICE_H_
#define PRIMITIV_C_DEVICES_OPENCL_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param platform_id Platform ID.
 * @param device_id Device ID on the selected platform.
 * @param rng_seed Seed value of the random number generator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateOpenCLDevice(
    uint32_t platform_id, uint32_t device_id, primitivDevice_t **newobj);

/**
 * Creates a new Device object.
 * @param platform_id Platform ID.
 * @param device_id Device ID on the selected platform.
 * @param rng_seed Seed value of the random number generator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateOpenCLDeviceWithSeed(
    uint32_t platform_id, uint32_t device_id, uint32_t rng_seed,
    primitivDevice_t **newobj);

/**
 * Retrieves the number of active platforms.
 * @param retval Pointer to receive the number of active platforms.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNumOpenCLPlatforms(
    uint32_t *retval);

/**
 * Retrieves the number of active devices on the specified platform.
 * @param platform_id Platform ID.
 *                    This value should be between 0 to num_platforms() - 1.
 * @param retval Pointer to receive the number of active devices.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNumOpenCLDevices(
    uint32_t platform_id, uint32_t *retval);

#endif  // PRIMITIV_C_DEVICES_OPENCL_DEVICE_H_
