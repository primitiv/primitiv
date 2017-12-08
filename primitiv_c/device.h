#ifndef PRIMITIV_C_DEVICE_H_
#define PRIMITIV_C_DEVICE_H_

#include "primitiv_c/define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Device primitiv_Device;

primitiv_Device *primitiv_Device_get_default();
primitiv_Device *safe_primitiv_Device_get_default(primitiv_Status *status);

void primitiv_Device_set_default(primitiv_Device *device);
void safe_primitiv_Device_set_default(primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_DEVICE_H_
