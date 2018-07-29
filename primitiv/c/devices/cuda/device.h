#ifndef PRIMITIV_C_DEVICES_CUDA_DEVICE_H_
#define PRIMITIV_C_DEVICES_CUDA_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 * @remarks The random number generator is initialized using
 *          `std::random_device`.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateCudaDevice(
    uint32_t device_id, primitivDevice_t **newobj);

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param rng_seed The seed value of the random number generator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateCudaDeviceWithSeed(
    uint32_t device_id, uint32_t rng_seed, primitivDevice_t **newobj);

/**
 * Retrieves the number of active hardwares.
 * @param retval Pointer to receive the number of active hardwares.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNumCudaDevices(
    uint32_t *retval);

#endif  // PRIMITIV_C_DEVICES_CUDA_DEVICE_H_
