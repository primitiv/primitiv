#ifndef PRIMITIV_C_DEVICE_H_
#define PRIMITIV_C_DEVICE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Device primitiv_Device;

primitiv_Device *primitiv_Device_get_default();

void primitiv_Device_set_default(primitiv_Device *device);

void primitiv_Device_delete(primitiv_Device *device);

primitiv_Device* primitiv_Naive_new();

primitiv_Device* primitiv_Naive_new_with_seed(uint32_t seed);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_DEVICE_H_
