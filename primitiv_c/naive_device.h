#ifndef PRIMITIV_C_NAIVE_DEVICE_H_
#define PRIMITIV_C_NAIVE_DEVICE_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Device *primitiv_Naive_new();
primitiv_Device *safe_primitiv_Naive_new(primitiv_Status *status);

primitiv_Device *primitiv_Naive_new_with_seed(uint32_t seed);
primitiv_Device *safe_primitiv_Naive_new_with_seed(uint32_t seed, primitiv_Status *status);

void primitiv_Naive_delete(primitiv_Device *device);
void safe_primitiv_Naive_delete(primitiv_Device *device, primitiv_Status *status);

void primitiv_Naive_dump_description(const primitiv_Device *device);
void safe_primitiv_Naive_dump_description(const primitiv_Device *device, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_NAIVE_DEVICE_H_
