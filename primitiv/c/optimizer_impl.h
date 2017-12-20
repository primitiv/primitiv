/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/optimizer.h>
#include <primitiv/c/status.h>

#define DECL_CONFIGS(name) \
CAPI extern uint32_t _CONCAT(name, _get_size_of_int_configs)( \
    const primitiv_Optimizer *optimizer); \
CAPI extern primitiv_Status _CONCAT(name, _get_int_configs)( \
    const primitiv_Optimizer *optimizer, char **keys, uint32_t *values, \
    size_t n); \
CAPI extern primitiv_Status _CONCAT(name, _set_int_configs)( \
    primitiv_Optimizer *optimizer, char **keys, uint32_t *values, size_t n); \
CAPI extern uint32_t _CONCAT(name, _get_size_of_float_configs)( \
    const primitiv_Optimizer *optimizer); \
CAPI extern primitiv_Status _CONCAT(name, _get_float_configs)( \
    const primitiv_Optimizer *optimizer, char **keys, float *values, \
    size_t n); \
CAPI extern primitiv_Status _CONCAT(name, _set_float_configs)( \
    primitiv_Optimizer *optimizer, char **keys, float *values, size_t n);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new (SGD) Optimizer object.
 * @param eta Learning rate.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_SGD_new(float eta);

/**
 * Returns the learning rate.
 * @param optimizer Pointer of a handler.
 * @return Learning rate.
 */
CAPI extern float primitiv_optimizers_SGD_eta(
    const primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_optimizers_SGD configs

/**
 * Creates a new (MomentumSGD) Optimizer object.
 * @param eta Learning rate.
 * @param momentum Decay factor of the momentum.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_MomentumSGD_new(
    float eta, float momentum);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @return The value of eta.
 */
CAPI extern float primitiv_optimizers_MomentumSGD_eta(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter momentum.
 * @param optimizer Pointer of a handler.
 * @return The value of momentum.
 */
CAPI extern float primitiv_optimizers_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_MementumSGD configs

/**
 * Creates a new (AdaGrad) Optimizer object.
 * @param eta Learning rate.
 * @param eps Bias of power.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_AdaGrad_new(float eta,
                                                                float eps);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @return The value of eta.
 */
CAPI extern float primitiv_optimizers_AdaGrad_eta(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @return The value of eps.
 */
CAPI extern float primitiv_optimizers_AdaGrad_eps(
    const primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_optimizers_AdaGrad configs

/**
 * Creates a new (RMSProp) Optimizer object.
 * @param eta Learning rate.
 * @param alpha Decay factor of moment.
 * @param eps Bias of power.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps);

/**
 * Returns the hyperparameter eta.
 * @param optimizer Pointer of a handler.
 * @return The value of eta.
 */
CAPI extern float primitiv_optimizers_RMSProp_eta(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter alpha.
 * @param optimizer Pointer of a handler.
 * @return The value of alpha.
 */
CAPI extern float primitiv_optimizers_RMSProp_alpha(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @return The value of eps.
 */
CAPI extern float primitiv_optimizers_RMSProp_eps(
    const primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_optimizers_RMSProp configs

/**
 * Creates a new (AdaDelta) Optimizer object.
 * @param rho Decay factor of RMS operation.
 * @param eps Bias of RMS values.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_AdaDelta_new(float rho,
                                                                 float eps);

/**
 * Returns the hyperparameter rho.
 * @param optimizer Pointer of a handler.
 * @return The value of rho.
 */
CAPI extern float primitiv_optimizers_AdaDelta_rho(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @return The value of eps.
 */
CAPI extern float primitiv_optimizers_AdaDelta_eps(
    const primitiv_Optimizer *optimizer);

// @TODO: Implement getter/setter of primitiv_optimizers_AdaDelta configs

/**
 * Creates a new Adam object.
 * @param alpha Learning rate.
 * @param beta1 Decay factor of momentum history.
 * @param beta2 Decay factor of power history.
 * @param eps Bias of power.
 * @return Pointer of a handler.
 */
CAPI extern primitiv_Optimizer *primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps);

/**
 * Returns the hyperparameter alpha.
 * @param optimizer Pointer of a handler.
 * @return The value of alpha.
 */
CAPI extern float primitiv_optimizers_Adam_alpha(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter beta1.
 * @param optimizer Pointer of a handler.
 * @return The value of beta1.
 */
CAPI extern float primitiv_optimizers_Adam_beta1(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter beta2.
 * @param optimizer Pointer of a handler.
 * @return The value of beta2.
 */
CAPI extern float primitiv_optimizers_Adam_beta2(
    const primitiv_Optimizer *optimizer);

/**
 * Returns the hyperparameter eps.
 * @param optimizer Pointer of a handler.
 * @return The value of eps.
 */
CAPI extern float primitiv_optimizers_Adam_eps(
    const primitiv_Optimizer *optimizer);

#ifdef __cplusplus
}  // end extern "C"
#endif

#undef DECL_CONFIGS

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
