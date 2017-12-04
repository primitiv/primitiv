#ifndef PRIMITIV_C_NAIVE_DEVICE_H_
#define PRIMITIV_C_NAIVE_DEVICE_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Device *primitiv_Naive_new();

primitiv_Device *primitiv_Naive_new_with_seed(uint32_t seed);

void primitiv_Naive_delete(primitiv_Device *device);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
