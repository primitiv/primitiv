#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/optimizer.h>

/**
 * Creates a new (SGD) Optimizer object.
 * @param eta Learning rate.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_SGD_new(
    float eta, primitiv_Optimizer **optimizer);

/**
 * Creates a new (MomentumSGD) Optimizer object.
 * @param eta Learning rate.
 * @param momentum Decay factor of the momentum.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_MomentumSGD_new(
    float eta, float momentum, primitiv_Optimizer **optimizer);

/**
 * Creates a new (AdaGrad) Optimizer object.
 * @param eta Learning rate.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_AdaGrad_new(
    float eta, float eps, primitiv_Optimizer **optimizer);

/**
 * Creates a new (RMSProp) Optimizer object.
 * @param eta Learning rate.
 * @param alpha Decay factor of moment.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps, primitiv_Optimizer **optimizer);

/**
 * Creates a new (AdaDelta) Optimizer object.
 * @param rho Decay factor of RMS operation.
 * @param eps Bias of RMS values.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_AdaDelta_new(
    float rho, float eps, primitiv_Optimizer **optimizer);

/**
 * Creates a new Adam object.
 * @param alpha Learning rate.
 * @param beta1 Decay factor of momentum history.
 * @param beta2 Decay factor of power history.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
PRIMITIV_C_API PRIMITIV_C_STATUS primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps,
    primitiv_Optimizer **optimizer);

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
