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

/**
 * Creates a new Initializer object that generates a same-value tensor.
 * @param k Constant value to fill a tensor.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_Constant_new(float k);

/**
 * Deletes the Constant Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_Constant_delete(primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Constant_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

/**
 * Creates a new Initializer object that uses a parameterized uniform
 * distribution (lower, upper].
 * @param lower Lower boundary of the uniform distribution.
 * @param upper Upper boundary of the uniform distribution.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_Uniform_new(float lower,
                                                       float upper);

/**
 * Deletes the Uniform Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_Uniform_delete(primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Uniform_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

/**
 * Creates a new Initializer object that uses a parameterized normal
 * distribution N(mean, sd).
 * @param mean Mean of the normal distribution.
 * @param sd Standard deviation of the normal distribution.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_Normal_new(float mean, float sd);

/**
 * Deletes the Normal Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_Normal_delete(primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Normal_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

/**
 * Creates a new Initializer object that generates a identity tensor.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_Identity_new();

/**
 * Deletes the Identity Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_Identity_delete(primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_Identity_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the uniform distribution.
 * @param scale Constant value that determines the scale of the uniform
 *              distribution.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_XavierUniform_new(float scale);

/**
 * Deletes the XavierUniform Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_XavierUniform_delete(
    primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_XavierUniform_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the normal distribution.
 * @param scale Constant value that determines the scale of the normal
 *              distribution.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Initializer *primitiv_XavierNormal_new(float scale);

/**
 * Deletes the XavierNormal Initializer object.
 * @param initializer Pointer of a handler.
 */
CAPI extern void primitiv_XavierNormal_delete(
    primitiv_Initializer *initializer);

/**
 * Provides an initialized tensor.
 * @param initializer Pointer of a handler.
 * @param x Tensor object to be initialized.
 * @return Status code.
 */
CAPI extern primitiv_Status primitiv_XavierNormal_apply(
    const primitiv_Initializer *initializer, primitiv_Tensor *x);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
