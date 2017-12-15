/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include <primitiv/c/define.h>
#include <primitiv/c/optimizer.h>
#include <primitiv/c/status.h>
#include <primitiv/c/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

CAPI extern primitiv_Optimizer *primitiv_SGD_new(float eta);
CAPI extern primitiv_Optimizer *safe_primitiv_SGD_new(float eta,
                                                      primitiv_Status *status);

CAPI extern void primitiv_SGD_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_SGD_delete(primitiv_Optimizer *optimizer,
                                          primitiv_Status *status);

CAPI extern float primitiv_SGD_eta(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_SGD_eta(const primitiv_Optimizer *optimizer,
                                        primitiv_Status *status);

CAPI extern void primitiv_SGD_get_configs(const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_SGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_SGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_SGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern primitiv_Optimizer *primitiv_MomentumSGD_new(float eta,
                                                         float momentum);
CAPI extern primitiv_Optimizer *safe_primitiv_MomentumSGD_new(
    float eta, float momentum, primitiv_Status *status);

CAPI extern void primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer,
                                                  primitiv_Status *status);

CAPI extern float primitiv_MomentumSGD_eta(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_MomentumSGD_eta(
    const primitiv_Optimizer *optimizer, primitiv_Status *status);

CAPI extern float primitiv_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer, primitiv_Status *status);

CAPI extern void primitiv_MomentumSGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_MomentumSGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_MomentumSGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_MomentumSGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern primitiv_Optimizer *primitiv_AdaGrad_new(float eta, float eps);
CAPI extern primitiv_Optimizer *safe_primitiv_AdaGrad_new(
    float eta, float eps, primitiv_Status *status);

CAPI extern void primitiv_AdaGrad_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_AdaGrad_delete(primitiv_Optimizer *optimizer,
                                              primitiv_Status *status);

CAPI extern float primitiv_AdaGrad_eta(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_AdaGrad_eta(const primitiv_Optimizer *optimizer,
                                            primitiv_Status *status);

CAPI extern float primitiv_AdaGrad_eps(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_AdaGrad_eps(const primitiv_Optimizer *optimizer,
                                            primitiv_Status *status);

CAPI extern void primitiv_AdaGrad_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_AdaGrad_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_AdaGrad_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_AdaGrad_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern primitiv_Optimizer *primitiv_RMSProp_new(float eta,
                                                     float alpha,
                                                     float eps);
CAPI extern primitiv_Optimizer *safe_primitiv_RMSProp_new(
    float eta, float alpha, float eps, primitiv_Status *status);

CAPI extern void primitiv_RMSProp_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_RMSProp_delete(primitiv_Optimizer *optimizer,
                                              primitiv_Status *status);

CAPI extern float primitiv_RMSProp_eta(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_RMSProp_eta(const primitiv_Optimizer *optimizer,
                                            primitiv_Status *status);

CAPI extern float primitiv_RMSProp_alpha(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_RMSProp_alpha(
    const primitiv_Optimizer *optimizer, primitiv_Status *status);

CAPI extern float primitiv_RMSProp_eps(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_RMSProp_eps(const primitiv_Optimizer *optimizer,
                                            primitiv_Status *status);

CAPI extern void primitiv_RMSProp_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_RMSProp_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_RMSProp_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_RMSProp_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern primitiv_Optimizer *primitiv_AdaDelta_new(float rho, float eps);
CAPI extern primitiv_Optimizer *safe_primitiv_AdaDelta_new(
    float rho, float eps, primitiv_Status *status);

CAPI extern void primitiv_AdaDelta_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_AdaDelta_delete(primitiv_Optimizer *optimizer,
                                               primitiv_Status *status);

CAPI extern float primitiv_AdaDelta_rho(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_AdaDelta_rho(
    const primitiv_Optimizer *optimizer, primitiv_Status *status);

CAPI extern float primitiv_AdaDelta_eps(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_AdaDelta_eps(
    const primitiv_Optimizer *optimizer, primitiv_Status *status);

CAPI extern void primitiv_AdaDelta_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_AdaDelta_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_AdaDelta_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_AdaDelta_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern primitiv_Optimizer *primitiv_Adam_new(float alpha,
                                                  float beta1,
                                                  float beta2,
                                                  float eps);
CAPI extern primitiv_Optimizer *safe_primitiv_Adam_new(float alpha,
                                                       float beta1,
                                                       float beta2,
                                                       float eps,
                                                       primitiv_Status *status);

CAPI extern void primitiv_Adam_delete(primitiv_Optimizer *optimizer);
CAPI extern void safe_primitiv_Adam_delete(primitiv_Optimizer *optimizer,
                                           primitiv_Status *status);

CAPI extern float primitiv_Adam_alpha(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_Adam_alpha(const primitiv_Optimizer *optimizer,
                                           primitiv_Status *status);

CAPI extern float primitiv_Adam_beta1(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_Adam_beta1(const primitiv_Optimizer *optimizer,
                                           primitiv_Status *status);

CAPI extern float primitiv_Adam_beta2(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_Adam_beta2(const primitiv_Optimizer *optimizer,
                                           primitiv_Status *status);

CAPI extern float primitiv_Adam_eps(const primitiv_Optimizer *optimizer);
CAPI extern float safe_primitiv_Adam_eps(const primitiv_Optimizer *optimizer,
                                         primitiv_Status *status);

CAPI extern void primitiv_Adam_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_Adam_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

CAPI extern void primitiv_Adam_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs);
CAPI extern void safe_primitiv_Adam_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
