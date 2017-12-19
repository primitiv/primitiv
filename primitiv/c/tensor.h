/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_TENSOR_H_
#define PRIMITIV_C_TENSOR_H_

#include <primitiv/c/define.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct primitiv_Tensor primitiv_Tensor;

CAPI extern primitiv_Tensor *primitiv_Tensor_new();
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_new(primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_new_from_tensor(
    primitiv_Tensor *tensor);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_new_from_tensor(
    primitiv_Tensor *tensor, primitiv_Status *status);

CAPI extern void primitiv_Tensor_delete(primitiv_Tensor *tensor);
CAPI extern void safe_primitiv_Tensor_delete(primitiv_Tensor *tensor,
                                             primitiv_Status *status);

CAPI extern bool primitiv_Tensor_valid(const primitiv_Tensor *tensor);
CAPI extern bool safe_primitiv_Tensor_valid(const primitiv_Tensor *tensor,
                                            primitiv_Status *status);

CAPI extern primitiv_Shape *primitiv_Tensor_shape(
    const primitiv_Tensor *tensor);
CAPI extern primitiv_Shape *safe_primitiv_Tensor_shape(
    const primitiv_Tensor *tensor, primitiv_Status *status);

CAPI extern primitiv_Device *primitiv_Tensor_device(
    const primitiv_Tensor *tensor);
CAPI extern primitiv_Device *safe_primitiv_Tensor_device(
    const primitiv_Tensor *tensor, primitiv_Status *status);

CAPI extern float primitiv_Tensor_to_float(const primitiv_Tensor *tensor);
CAPI extern float safe_primitiv_Tensor_to_float(const primitiv_Tensor *tensor,
                                                primitiv_Status *status);

CAPI extern void primitiv_Tensor_to_array(const primitiv_Tensor *tensor,
                                          float *array);
CAPI extern void safe_primitiv_Tensor_to_array(const primitiv_Tensor *tensor,
                                               float *array,
                                               primitiv_Status *status);

CAPI extern uint32_t *primitiv_Tensor_argmax(const primitiv_Tensor *tensor,
                                             uint32_t dim);
CAPI extern uint32_t *safe_primitiv_Tensor_argmax(const primitiv_Tensor *tensor,
                                                  uint32_t dim,
                                                  primitiv_Status *status);

CAPI extern uint32_t *primitiv_Tensor_argmin(const primitiv_Tensor *tensor,
                                             uint32_t dim);
CAPI extern uint32_t *safe_primitiv_Tensor_argmin(const primitiv_Tensor *tensor,
                                                  uint32_t dim,
                                                  primitiv_Status *status);

CAPI extern void primitiv_Tensor_reset(primitiv_Tensor *tensor, float k);
CAPI extern void safe_primitiv_Tensor_reset(primitiv_Tensor *tensor,
                                            float k,
                                            primitiv_Status *status);

CAPI extern void primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                                const float *values);
CAPI extern void safe_primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                                     const float *values,
                                                     primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_reshape(
    const primitiv_Tensor *tensor, const primitiv_Shape *new_shape);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_reshape(
    const primitiv_Tensor *tensor,
    const primitiv_Shape *new_shape,
    primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_flatten(
    const primitiv_Tensor *tensor);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_flatten(
    const primitiv_Tensor *tensor, primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_inplace_multiply_const(
    primitiv_Tensor *tensor, float k);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_inplace_multiply_const(
    primitiv_Tensor *tensor, float k, primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_inplace_add(
    primitiv_Tensor *tensor, const primitiv_Tensor *x);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_inplace_add(
    primitiv_Tensor *tensor, const primitiv_Tensor *x, primitiv_Status *status);

CAPI extern primitiv_Tensor *primitiv_Tensor_inplace_subtract(
    primitiv_Tensor *tensor, const primitiv_Tensor *x);
CAPI extern primitiv_Tensor *safe_primitiv_Tensor_inplace_subtract(
    primitiv_Tensor *tensor, const primitiv_Tensor *x, primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_TENSOR_H_
