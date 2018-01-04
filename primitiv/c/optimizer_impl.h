#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/optimizer.h>
#include <primitiv/c/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new (SGD) Optimizer object.
 * @param eta Learning rate.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_SGD_new(
    float eta, primitiv_Optimizer **optimizer);

/**
 * Returns the learning rate.
 * @param optimizer Pointer of a handler.
 * @param eta Pointer to receive the learning rate.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_SGD_eta(
    const primitiv_Optimizer *optimizer, float *eta);

/**
 * Creates a new (MomentumSGD) Optimizer object.
 * @param eta Learning rate.
 * @param momentum Decay factor of the momentum.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_MomentumSGD_new(
    float eta, float momentum, primitiv_Optimizer **optimizer);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @param eta Pointer to receive the value of eta.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_MomentumSGD_eta(
    const primitiv_Optimizer *optimizer, float *eta);

/**
 * Returns the hyperparameter momentum.
 * @param optimizer Pointer of a handler.
 * @param momentum Pointer to receive the value of momentum.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer, float *momentum);

/**
 * Creates a new (AdaGrad) Optimizer object.
 * @param eta Learning rate.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaGrad_new(
    float eta, float eps, primitiv_Optimizer **optimizer);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @param eta Pointer to receive the value of eta.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaGrad_eta(
    const primitiv_Optimizer *optimizer, float *eta);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @param eps Pointer to receive the value of eps.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaGrad_eps(
    const primitiv_Optimizer *optimizer, float *eps);

/**
 * Creates a new (RMSProp) Optimizer object.
 * @param eta Learning rate.
 * @param alpha Decay factor of moment.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps, primitiv_Optimizer **optimizer);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @param eta Pointer to receive the value of eta.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_RMSProp_eta(
    const primitiv_Optimizer *optimizer, float *eta);

/**
 * Returns the hyperparameter alpha.
 * @param optimizer Pointer of a handler.
 * @param alpha Pointer to receive the value of alpha.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_RMSProp_alpha(
    const primitiv_Optimizer *optimizer, float *alpha);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @param eps Pointer to receive the value of eps.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_RMSProp_eps(
    const primitiv_Optimizer *optimizer, float *eps);

/**
 * Creates a new (AdaDelta) Optimizer object.
 * @param rho Decay factor of RMS operation.
 * @param eps Bias of RMS values.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaDelta_new(
    float rho, float eps, primitiv_Optimizer **optimizer);

/**
 * Returns the hyperparameter rho.
 * @param optimizer Pointer of a handler.
 * @param rho Pointer to receive the value of rho.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaDelta_rho(
    const primitiv_Optimizer *optimizer, float *rho);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @param eps Pointer to receive the value of eps.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_AdaDelta_eps(
    const primitiv_Optimizer *optimizer, float *eps);

/**
 * Creates a new Adam object.
 * @param alpha Learning rate.
 * @param beta1 Decay factor of momentum history.
 * @param beta2 Decay factor of power history.
 * @param eps Bias of power.
 * @param optimizer Pointer to receive a handler.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps,
    primitiv_Optimizer **optimizer);

/**
 * Returns the hyperparameter alpha.
 * @param optimizer Pointer of a handler.
 * @param alpha Pointer to receive the value of alpha.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_Adam_alpha(
    const primitiv_Optimizer *optimizer, float *alpha);

/**
 * Returns the hyperparameter beta1.
 * @param optimizer Pointer of a handler.
 * @param beta1 Pointer to receive the value of beta1.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_Adam_beta1(
    const primitiv_Optimizer *optimizer, float *beta1);

/**
 * Returns the hyperparameter beta2.
 * @param optimizer Pointer of a handler.
 * @param beta2 Pointer to receive the value of beta2.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_Adam_beta2(
    const primitiv_Optimizer *optimizer, float *beta2);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @param eps Pointer to receive the value of eps.
 * @return Status code.
 */
extern PRIMITIV_C_API primitiv_Status primitiv_optimizers_Adam_eps(
    const primitiv_Optimizer *optimizer, float *eps);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
