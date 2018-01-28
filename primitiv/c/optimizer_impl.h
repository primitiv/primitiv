#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/optimizer.h>

/**
 * Creates a new (SGD) Optimizer object.
 * @param eta Learning rate.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateSgdOptimizer(
    float eta, primitivOptimizer_t **newobj);

/**
 * Creates a new (MomentumSGD) Optimizer object.
 * @param eta Learning rate.
 * @param momentum Decay factor of the momentum.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateMomentumSgdOptimizer(
    float eta, float momentum, primitivOptimizer_t **newobj);

/**
 * Creates a new (AdaGrad) Optimizer object.
 * @param eta Learning rate.
 * @param eps Bias of power.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateAdaGradOptimizer(
    float eta, float eps, primitivOptimizer_t **newobj);

/**
 * Creates a new (RMSProp) Optimizer object.
 * @param eta Learning rate.
 * @param alpha Decay factor of moment.
 * @param eps Bias of power.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateRmsPropOptimizer(
    float eta, float alpha, float eps, primitivOptimizer_t **newobj);

/**
 * Creates a new (AdaDelta) Optimizer object.
 * @param rho Decay factor of RMS operation.
 * @param eps Bias of RMS values.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateAdaDeltaOptimizer(
    float rho, float eps, primitivOptimizer_t **newobj);

/**
 * Creates a new Adam object.
 * @param alpha Learning rate.
 * @param beta1 Decay factor of momentum history.
 * @param beta2 Decay factor of power history.
 * @param eps Bias of power.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitivCreateAdamOptimizer(
    float alpha, float beta1, float beta2, float eps,
    primitivOptimizer_t **newobj);

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
