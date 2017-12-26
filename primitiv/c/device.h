/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_DEVICE_H_
#define PRIMITIV_C_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque type of Device.
 */
typedef struct primitiv_Device primitiv_Device;

/**
 * Retrieves the current default device.
 * @param device Pointer to receive the current default device.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_Device_get_default(
    primitiv_Device **device);

/**
 * Specifies a new default device.
 * @param device Pointer of the new default device.
 */
extern PRIMITIV_C_API void primitiv_Device_set_default(primitiv_Device *device);

/**
 * Deletes the Device object.
 * @param device Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Device_delete(primitiv_Device *device);

/**
 * Prints device description to stderr.
 * @param device Pointer of a handler.
 */
extern PRIMITIV_C_API void primitiv_Device_dump_description(
    const primitiv_Device *device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_DEVICE_H_
