#ifndef PRIMITIV_C_INITIALIZER_IMPL_H_
#define PRIMITIV_C_INITIALIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/initializer.h>
#include <primitiv/c/tensor.h>

/**
 * Creates a new Initializer object that generates a same-value tensor.
 * @param k Constant value to fill a tensor.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_Constant_new(
    float k, primitivInitializer_t **initializer);

/**
 * Creates a new Initializer object that uses a parameterized uniform
 * distribution (lower, upper].
 * @param lower Lower boundary of the uniform distribution.
 * @param upper Upper boundary of the uniform distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_Uniform_new(
    float lower, float upper, primitivInitializer_t **initializer);

/**
 * Creates a new Initializer object that uses a parameterized normal
 * distribution N(mean, sd).
 * @param mean Mean of the normal distribution.
 * @param sd Standard deviation of the normal distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_Normal_new(
    float mean, float sd, primitivInitializer_t **initializer);

/**
 * Creates a new Initializer object that generates a identity tensor.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_Identity_new(
    primitivInitializer_t **initializer);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the uniform distribution.
 * @param scale Constant value that determines the scale of the uniform
 *              distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_XavierUniform_new(
    float scale, primitivInitializer_t **initializer);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the normal distribution.
 * @param scale Constant value that determines the scale of the normal
 *              distribution.
 * @param initializer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_initializers_XavierNormal_new(
    float scale, primitivInitializer_t **initializer);

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
