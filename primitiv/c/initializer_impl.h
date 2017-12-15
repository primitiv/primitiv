/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_INITIALIZER_IMPL_H_
#define PRIMITIV_C_INITIALIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/initializer.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

CAPI extern primitiv_Initializer *primitiv_Constant_new(float k);
CAPI extern primitiv_Initializer *safe_primitiv_Constant_new(
    float k, primitiv_Status *status);

CAPI extern void primitiv_Constant_delete(primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_Constant_delete(
    primitiv_Initializer *initializer, primitiv_Status *status);

CAPI extern void primitiv_Constant_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);
CAPI extern void safe_primitiv_Constant_apply(
    const primitiv_Initializer *initializer,
    primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Initializer *primitiv_Uniform_new(float lower,
                                                       float upper);
CAPI extern primitiv_Initializer *safe_primitiv_Uniform_new(
    float lower, float upper, primitiv_Status *status);

CAPI extern void primitiv_Uniform_delete(primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_Uniform_delete(primitiv_Initializer *initializer,
                                              primitiv_Status *status);

CAPI extern void primitiv_Uniform_apply(const primitiv_Initializer *initializer,
                                        primitiv_Tensor *x);
CAPI extern void safe_primitiv_Uniform_apply(
    const primitiv_Initializer *initializer,
    primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Initializer *primitiv_Normal_new(float mean, float sd);
CAPI extern primitiv_Initializer *safe_primitiv_Normal_new(
    float mean, float sd, primitiv_Status *status);

CAPI extern void primitiv_Normal_delete(primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_Normal_delete(primitiv_Initializer *initializer,
                                             primitiv_Status *status);

CAPI extern void primitiv_Normal_apply(const primitiv_Initializer *initializer,
                                       primitiv_Tensor *x);
CAPI extern void safe_primitiv_Normal_apply(
    const primitiv_Initializer *initializer,
    primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Initializer *primitiv_Identity_new();
CAPI extern primitiv_Initializer *safe_primitiv_Identity_new(
    primitiv_Status *status);

CAPI extern void primitiv_Identity_delete(primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_Identity_delete(
    primitiv_Initializer *initializer, primitiv_Status *status);

CAPI extern void primitiv_Identity_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);
CAPI extern void safe_primitiv_Identity_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Initializer *primitiv_XavierUniform_new(float scale);
CAPI extern primitiv_Initializer *safe_primitiv_XavierUniform_new(
    float scale, primitiv_Status *status);

CAPI extern void primitiv_XavierUniform_delete(
    primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_XavierUniform_delete(
    primitiv_Initializer *initializer, primitiv_Status *status);

CAPI extern void primitiv_XavierUniform_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);
CAPI extern void safe_primitiv_XavierUniform_apply(
    const primitiv_Initializer *initializer,
    primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Initializer *primitiv_XavierNormal_new(float scale);
CAPI extern primitiv_Initializer *safe_primitiv_XavierNormal_new(
    float scale, primitiv_Status *status);

CAPI extern void primitiv_XavierNormal_delete(
    primitiv_Initializer *initializer);
CAPI extern void safe_primitiv_XavierNormal_delete(
    primitiv_Initializer *initializer, primitiv_Status *status);

CAPI extern void primitiv_XavierNormal_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);
CAPI extern void safe_primitiv_XavierNormal_apply(
    const primitiv_Initializer *initializer,
    primitiv_Tensor *x,
    primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
