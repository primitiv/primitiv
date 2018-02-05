#ifndef PRIMITIV_C_DEVICE_H_
#define PRIMITIV_C_DEVICE_H_

#include <primitiv/c/define.h>

/**
 * Opaque type of Device.
 */
typedef struct primitivDevice primitivDevice_t;

/**
 * Retrieves the current default device.
 * @param retval Pointer to receive the current default device.
 * @return Status code.
 * @remarks The pointer returned by this function is owned by the library, and
 *          should not be passed to `primitiv_Device_delete()`.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivGetDefaultDevice(
    primitivDevice_t **retval);

/**
 * Specifies a new default device.
 * @param device Pointer of the new default device.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivSetDefaultDevice(
    primitivDevice_t *device);

/**
 * Deletes the Device object.
 * @param device Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDeleteDevice(
    primitivDevice_t *device);

/**
 * Prints device description to stderr.
 * @param device Pointer of a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivDumpDeviceDescription(
    const primitivDevice_t *device);

#endif  // PRIMITIV_C_DEVICE_H_
