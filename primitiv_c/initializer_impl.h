#ifndef PRIMITIV_C_INITIALIZER_IMPL_H_
#define PRIMITIV_C_INITIALIZER_IMPL_H_

#include "primitiv_c/define.h"
#include "primitiv_c/initializer.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Initializer *primitiv_Constant_new(float k);

void primitiv_Constant_delete(primitiv_Initializer *initializer);

primitiv_Initializer *primitiv_Uniform_new(float lower, float upper);

void primitiv_Uniform_delete(primitiv_Initializer *initializer);

primitiv_Initializer *primitiv_Normal_new(float mean, float sd);

void primitiv_Normal_delete(primitiv_Initializer *initializer);

primitiv_Initializer *primitiv_Identity_new();

void primitiv_Identity_delete(primitiv_Initializer *initializer);

primitiv_Initializer *primitiv_XavierUniform_new(float scale);

void primitiv_XavierUniform_delete(primitiv_Initializer *initializer);

primitiv_Initializer *primitiv_XavierNormal_new(float scale);

void primitiv_XavierNormal_delete(primitiv_Initializer *initializer);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
