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
extern PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_devices_CUDA_new(
    uint32_t device_id, primitiv_Device **device);

/**
 * Creates a new Device object.
 * @param device_id ID of the physical GPU.
 * @param rng_seed The seed value of the random number generator.
 * @param device Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_devices_CUDA_new_with_seed(
    uint32_t device_id, uint32_t rng_seed, primitiv_Device **device);

/**
 * Retrieves the number of active hardwares.
 * @param num_devices Pointer to receive the number of active hardwares.
 * @return Status code.
 */
extern PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_devices_CUDA_num_devices(
    uint32_t *num_devices);

#endif  // PRIMITIV_C_CUDA_DEVICE_H_
