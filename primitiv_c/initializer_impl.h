#ifndef PRIMITIV_C_INITIALIZER_IMPL_H_
#define PRIMITIV_C_INITIALIZER_IMPL_H_

#include "primitiv_c/define.h"
#include "primitiv_c/initializer.h"
#include "primitiv_c/status.h"
#include "primitiv_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Initializer *primitiv_Constant_new(float k);
primitiv_Initializer *safe_primitiv_Constant_new(float k, primitiv_Status *status);

void primitiv_Constant_delete(primitiv_Initializer *initializer);
void safe_primitiv_Constant_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_Constant_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_Constant_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

primitiv_Initializer *primitiv_Uniform_new(float lower, float upper);
primitiv_Initializer *safe_primitiv_Uniform_new(float lower, float upper, primitiv_Status *status);

void primitiv_Uniform_delete(primitiv_Initializer *initializer);
void safe_primitiv_Uniform_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_Uniform_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_Uniform_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

primitiv_Initializer *primitiv_Normal_new(float mean, float sd);
primitiv_Initializer *safe_primitiv_Normal_new(float mean, float sd, primitiv_Status *status);

void primitiv_Normal_delete(primitiv_Initializer *initializer);
void safe_primitiv_Normal_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_Normal_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_Normal_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

primitiv_Initializer *primitiv_Identity_new();
primitiv_Initializer *safe_primitiv_Identity_new(primitiv_Status *status);

void primitiv_Identity_delete(primitiv_Initializer *initializer);
void safe_primitiv_Identity_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_Identity_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_Identity_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

primitiv_Initializer *primitiv_XavierUniform_new(float scale);
primitiv_Initializer *safe_primitiv_XavierUniform_new(float scale, primitiv_Status *status);

void primitiv_XavierUniform_delete(primitiv_Initializer *initializer);
void safe_primitiv_XavierUniform_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_XavierUniform_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_XavierUniform_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

primitiv_Initializer *primitiv_XavierNormal_new(float scale);
primitiv_Initializer *safe_primitiv_XavierNormal_new(float scale, primitiv_Status *status);

void primitiv_XavierNormal_delete(primitiv_Initializer *initializer);
void safe_primitiv_XavierNormal_delete(primitiv_Initializer *initializer, primitiv_Status *status);

void primitiv_XavierNormal_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x);
void safe_primitiv_XavierNormal_apply(const primitiv_Initializer *initializer, primitiv_Tensor *x, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
