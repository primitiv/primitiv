#ifndef PRIMITIV_C_CUDA_DEVICE_H_
#define PRIMITIV_C_CUDA_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param device Pointer to receive a handler.
 * @return Status code.
 * @remarks The random number generator is initialized using
 *          `std::random_device`.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateCudaDevice(
    uint32_t device_id, primitivDevice_t **device);

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param rng_seed The seed value of the random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateCudaDeviceWithSeed(
    uint32_t device_id, uint32_t rng_seed, primitivDevice_t **device);

/**
 * Retrieves the number of active hardwares.
 * @param num_devices Pointer to receive the number of active hardwares.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetNumCudaDevices(
    uint32_t *num_devices);

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
