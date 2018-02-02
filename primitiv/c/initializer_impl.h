#ifndef PRIMITIV_C_INITIALIZER_IMPL_H_
#define PRIMITIV_C_INITIALIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/initializer.h>
#include <primitiv/c/tensor.h>

/**
 * Creates a new Initializer object that generates a same-value tensor.
 * @param k Constant value to fill a tensor.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateConstantInitializer(
    float k, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that uses a parameterized uniform
 * distribution (lower, upper].
 * @param lower Lower boundary of the uniform distribution.
 * @param upper Upper boundary of the uniform distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateUniformInitializer(
    float lower, float upper, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that uses a parameterized normal
 * distribution N(mean, sd).
 * @param mean Mean of the normal distribution.
 * @param sd Standard deviation of the normal distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateNormalInitializer(
    float mean, float sd, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that generates a identity tensor.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateIdentityInitializer(
    primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the uniform distribution.
 * @param scale Constant value that determines the scale of the uniform
 *              distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateXavierUniformInitializer(
    float scale, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the normal distribution.
 * @param scale Constant value that determines the scale of the normal
 *              distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateXavierNormalInitializer(
    float scale, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the uniform distribution for conv2d filters.
 * @param scale Constant value that determines the scale of the uniform
 *              distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateXavierUniformConv2DInitializer(
    float scale, primitivInitializer_t **newobj);

/**
 * Creates a new Initializer object that generates a tensor by the Xavier matrix
 * initialization with the normal distribution for conv2d filters.
 * @param scale Constant value that determines the scale of the normal
 *              distribution.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateXavierNormalConv2DInitializer(
    float scale, primitivInitializer_t **newobj);

#endif  // PRIMITIV_C_INITIALIZER_IMPL_H_
