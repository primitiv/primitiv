/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/optimizer_impl.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/optimizer_impl.h>

using primitiv::optimizers::SGD;
using primitiv::optimizers::MomentumSGD;
using primitiv::optimizers::AdaGrad;
using primitiv::optimizers::RMSProp;
using primitiv::optimizers::AdaDelta;
using primitiv::optimizers::Adam;

#define CAST_TO_CC_SGD(x) reinterpret_cast<SGD*>(x)
#define CAST_TO_CONST_CC_SGD(x) reinterpret_cast<const SGD*>(x)

#define CAST_TO_CC_MOMENTUM_SGD(x) reinterpret_cast<MomentumSGD*>(x)
#define CAST_TO_CONST_CC_MOMENTUM_SGD(x) reinterpret_cast<const MomentumSGD*>(x)

#define CAST_TO_CC_ADA_GRAD(x) reinterpret_cast<AdaGrad*>(x)
#define CAST_TO_CONST_CC_ADA_GRAD(x) reinterpret_cast<const AdaGrad*>(x)

#define CAST_TO_CC_RMS_PROP(x) reinterpret_cast<RMSProp*>(x)
#define CAST_TO_CONST_CC_RMS_PROP(x) reinterpret_cast<const RMSProp*>(x)

#define CAST_TO_CC_ADA_DELTA(x) reinterpret_cast<AdaDelta*>(x)
#define CAST_TO_CONST_CC_ADA_DELTA(x) reinterpret_cast<const AdaDelta*>(x)

#define CAST_TO_CC_ADAM(x) reinterpret_cast<Adam*>(x)
#define CAST_TO_CONST_CC_ADAM(x) reinterpret_cast<const Adam*>(x)

extern "C" {

primitiv_Optimizer *primitiv_SGD_new(float eta) {
  return to_c(new SGD(eta));
}
primitiv_Optimizer *safe_primitiv_SGD_new(float eta, primitiv_Status *status) {
  SAFE_RETURN(primitiv_SGD_new(eta), status, nullptr);
}

void primitiv_SGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_SGD(optimizer);
}
void safe_primitiv_SGD_delete(primitiv_Optimizer *optimizer,
                              primitiv_Status *status) {
  SAFE_EXPR(primitiv_SGD_delete(optimizer), status);
}

float primitiv_SGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_SGD(optimizer)->eta();
}
float safe_primitiv_SGD_eta(const primitiv_Optimizer *optimizer,
                            primitiv_Status *status) {
  SAFE_RETURN(primitiv_SGD_eta(optimizer), status, 0.0);
}

void primitiv_SGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_SGD(optimizer)->get_configs(*to_cc(uint_configs),
                                               *to_cc(float_configs));
}
void safe_primitiv_SGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_SGD_get_configs(optimizer, uint_configs, float_configs), status);
}

void primitiv_SGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_SGD(optimizer)->set_configs(*to_cc(uint_configs),
                                         *to_cc(float_configs));
}
void safe_primitiv_SGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_SGD_set_configs(optimizer, uint_configs, float_configs), status);
}

primitiv_Optimizer *primitiv_MomentumSGD_new(float eta, float momentum) {
  return to_c(new MomentumSGD(eta, momentum));
}
primitiv_Optimizer *safe_primitiv_MomentumSGD_new(float eta,
                                                  float momentum,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_MomentumSGD_new(eta, momentum), status, nullptr);
}

void primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_MOMENTUM_SGD(optimizer);
}
void safe_primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer,
                                      primitiv_Status *status) {
  SAFE_EXPR(primitiv_MomentumSGD_delete(optimizer), status);
}

float primitiv_MomentumSGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->eta();
}
float safe_primitiv_MomentumSGD_eta(const primitiv_Optimizer *optimizer,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_MomentumSGD_eta(optimizer), status, 0.0);
}

float primitiv_MomentumSGD_momentum(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->momentum();
}
float safe_primitiv_MomentumSGD_momentum(const primitiv_Optimizer *optimizer,
                                         primitiv_Status *status) {
  SAFE_RETURN(primitiv_MomentumSGD_momentum(optimizer), status, 0.0);
}

void primitiv_MomentumSGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->get_configs(*to_cc(uint_configs),
                                                        *to_cc(float_configs));
}
void safe_primitiv_MomentumSGD_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_MomentumSGD_get_configs(optimizer, uint_configs, float_configs),
      status);
}

void primitiv_MomentumSGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_MOMENTUM_SGD(optimizer)->set_configs(*to_cc(uint_configs),
                                                  *to_cc(float_configs));
}
void safe_primitiv_MomentumSGD_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_MomentumSGD_set_configs(optimizer, uint_configs, float_configs),
      status);
}

primitiv_Optimizer *primitiv_AdaGrad_new(float eta, float eps) {
  return to_c(new AdaGrad(eta, eps));
}
primitiv_Optimizer *safe_primitiv_AdaGrad_new(float eta,
                                              float eps,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaGrad_new(eta, eps), status, nullptr);
}

void primitiv_AdaGrad_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADA_GRAD(optimizer);
}
void safe_primitiv_AdaGrad_delete(primitiv_Optimizer *optimizer,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_AdaGrad_delete(optimizer), status);
}

float primitiv_AdaGrad_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
}
float safe_primitiv_AdaGrad_eta(const primitiv_Optimizer *optimizer,
                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaGrad_eta(optimizer), status, 0.0);
}

float primitiv_AdaGrad_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eps();
}
float safe_primitiv_AdaGrad_eps(const primitiv_Optimizer *optimizer,
                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaGrad_eps(optimizer), status, 0.0);
}

void primitiv_AdaGrad_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_ADA_GRAD(optimizer)->get_configs(*to_cc(uint_configs),
                                                    *to_cc(float_configs));
}
void safe_primitiv_AdaGrad_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_AdaGrad_get_configs(optimizer, uint_configs, float_configs),
      status);
}

void primitiv_AdaGrad_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_ADA_GRAD(optimizer)->set_configs(*to_cc(uint_configs),
                                              *to_cc(float_configs));
}
void safe_primitiv_AdaGrad_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_AdaGrad_set_configs(optimizer, uint_configs, float_configs),
      status);
}

primitiv_Optimizer *primitiv_RMSProp_new(float eta, float alpha, float eps) {
  return to_c(new RMSProp(eta, alpha, eps));
}
primitiv_Optimizer *safe_primitiv_RMSProp_new(float eta,
                                              float alpha,
                                              float eps,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_RMSProp_new(eta, alpha, eps), status, nullptr);
}

void primitiv_RMSProp_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_RMS_PROP(optimizer);
}
void safe_primitiv_RMSProp_delete(primitiv_Optimizer *optimizer,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_RMSProp_delete(optimizer), status);
}

float primitiv_RMSProp_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
}
float safe_primitiv_RMSProp_eta(const primitiv_Optimizer *optimizer,
                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_RMSProp_eta(optimizer), status, 0.0);
}

float primitiv_RMSProp_alpha(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_RMS_PROP(optimizer)->alpha();
}
float safe_primitiv_RMSProp_alpha(const primitiv_Optimizer *optimizer,
                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_RMSProp_alpha(optimizer), status, 0.0);
}

float primitiv_RMSProp_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_RMS_PROP(optimizer)->eps();
}
float safe_primitiv_RMSProp_eps(const primitiv_Optimizer *optimizer,
                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_RMSProp_eps(optimizer), status, 0.0);
}

void primitiv_RMSProp_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_RMS_PROP(optimizer)->get_configs(*to_cc(uint_configs),
                                                    *to_cc(float_configs));
}
void safe_primitiv_RMSProp_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_RMSProp_get_configs(optimizer, uint_configs, float_configs),
      status);
}

void primitiv_RMSProp_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_RMS_PROP(optimizer)->set_configs(*to_cc(uint_configs),
                                              *to_cc(float_configs));
}
void safe_primitiv_RMSProp_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_RMSProp_set_configs(optimizer, uint_configs, float_configs),
      status);
}

primitiv_Optimizer *primitiv_AdaDelta_new(float rho, float eps) {
  return to_c(new AdaDelta(rho, eps));
}
primitiv_Optimizer *safe_primitiv_AdaDelta_new(float rho,
                                               float eps,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaDelta_new(rho, eps), status, nullptr);
}

void primitiv_AdaDelta_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADA_DELTA(optimizer);
}
void safe_primitiv_AdaDelta_delete(primitiv_Optimizer *optimizer,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_AdaDelta_delete(optimizer), status);
}

float primitiv_AdaDelta_rho(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_DELTA(optimizer)->rho();
}
float safe_primitiv_AdaDelta_rho(const primitiv_Optimizer *optimizer,
                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaDelta_rho(optimizer), status, 0.0);
}

float primitiv_AdaDelta_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_DELTA(optimizer)->eps();
}
float safe_primitiv_AdaDelta_eps(const primitiv_Optimizer *optimizer,
                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_AdaDelta_eps(optimizer), status, 0.0);
}

void primitiv_AdaDelta_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_ADA_DELTA(optimizer)->get_configs(*to_cc(uint_configs),
                                                     *to_cc(float_configs));
}
void safe_primitiv_AdaDelta_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_AdaDelta_get_configs(optimizer, uint_configs, float_configs),
      status);
}

void primitiv_AdaDelta_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_ADA_DELTA(optimizer)->set_configs(*to_cc(uint_configs),
                                               *to_cc(float_configs));
}
void safe_primitiv_AdaDelta_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_AdaDelta_set_configs(optimizer, uint_configs, float_configs),
      status);
}

primitiv_Optimizer *primitiv_Adam_new(float alpha,
                                      float beta1,
                                      float beta2,
                                      float eps) {
  return to_c(new Adam(alpha, beta1, beta2, eps));
}
primitiv_Optimizer *safe_primitiv_Adam_new(float alpha,
                                           float beta1,
                                           float beta2,
                                           float eps,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Adam_new(alpha, beta1, beta2, eps), status, nullptr);
}

void primitiv_Adam_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADAM(optimizer);
}
void safe_primitiv_Adam_delete(primitiv_Optimizer *optimizer,
                               primitiv_Status *status) {
  SAFE_EXPR(primitiv_Adam_delete(optimizer), status);
}

float primitiv_Adam_alpha(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->alpha();
}
float safe_primitiv_Adam_alpha(const primitiv_Optimizer *optimizer,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Adam_alpha(optimizer), status, 0.0);
}

float primitiv_Adam_beta1(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->beta1();
}
float safe_primitiv_Adam_beta1(const primitiv_Optimizer *optimizer,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Adam_beta1(optimizer), status, 0.0);
}

float primitiv_Adam_beta2(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->beta2();
}
float safe_primitiv_Adam_beta2(const primitiv_Optimizer *optimizer,
                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_Adam_beta2(optimizer), status, 0.0);
}

float primitiv_Adam_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->eps();
}
float safe_primitiv_Adam_eps(const primitiv_Optimizer *optimizer,
                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_Adam_eps(optimizer), status, 0.0);
}

void primitiv_Adam_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs) {
  CAST_TO_CONST_CC_ADAM(optimizer)->get_configs(*to_cc(uint_configs),
                                                *to_cc(float_configs));
}
void safe_primitiv_Adam_get_configs(
    const primitiv_Optimizer *optimizer,
    primitiv_StrIntMap *uint_configs,
    primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Adam_get_configs(optimizer, uint_configs, float_configs),
      status);
}

void primitiv_Adam_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs) {
  CAST_TO_CC_ADAM(optimizer)->set_configs(*to_cc(uint_configs),
                                          *to_cc(float_configs));
}
void safe_primitiv_Adam_set_configs(
    primitiv_Optimizer *optimizer,
    const primitiv_StrIntMap *uint_configs,
    const primitiv_StrFloatMap *float_configs,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Adam_set_configs(optimizer, uint_configs, float_configs),
      status);
}

}  // end extern "C"
