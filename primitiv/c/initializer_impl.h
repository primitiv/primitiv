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
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_Constant_new(
    float k, primitiv_Initializer **initializer);

/**
 * Creates a new Initializer object that uses a parameterized uniform
 * distribution (lower, upper].
 * @param lower Lower boundary of the uniform distribution.
 * @param upper Upper boundary of the uniform distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_Uniform_new(
    float lower, float upper, primitiv_Initializer **initializer);

/**
 * Creates a new Initializer object that uses a parameterized normal
 * distribution N(mean, sd).
 * @param mean Mean of the normal distribution.
 * @param sd Standard deviation of the normal distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_Normal_new(
    float mean, float sd, primitiv_Initializer **initializer);

/**
 * Creates a new Initializer object that generates a identity tensor.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_Identity_new(
    primitiv_Initializer **initializer);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the uniform distribution.
 * @param scale Constant value that determines the scale of the uniform
 *              distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_XavierUniform_new(
    float scale, primitiv_Initializer **initializer);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the normal distribution.
 * @param scale Constant value that determines the scale of the normal
 *              distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_initializers_XavierNormal_new(
    float scale, primitiv_Initializer **initializer);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
