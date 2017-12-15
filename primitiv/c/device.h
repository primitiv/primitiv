/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_DEVICE_H_
#define PRIMITIV_C_DEVICE_H_

#include <primitiv/c/define.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Device primitiv_Device;

CAPI extern primitiv_Device *primitiv_Device_get_default();
CAPI extern primitiv_Device *safe_primitiv_Device_get_default(
    primitiv_Status *status);

CAPI extern void primitiv_Device_set_default(
    primitiv_Device *device);
CAPI extern void safe_primitiv_Device_set_default(primitiv_Device *device,
                                                  primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_DEVICE_H_
