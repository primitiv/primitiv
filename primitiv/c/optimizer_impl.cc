/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/optimizer_impl.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/optimizer_impl.h>

using primitiv::optimizers::SGD;
using primitiv::optimizers::MomentumSGD;
using primitiv::optimizers::AdaGrad;
using primitiv::optimizers::RMSProp;
using primitiv::optimizers::AdaDelta;
using primitiv::optimizers::Adam;
using primitiv::c::internal::to_c_ptr;

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

primitiv_Optimizer *primitiv_optimizers_SGD_new(float eta) {
  return to_c_ptr(new SGD(eta));
}

void primitiv_optimizers_SGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_SGD(optimizer);
}

float primitiv_optimizers_SGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_SGD(optimizer)->eta();
}

primitiv_Optimizer *primitiv_optimizers_MomentumSGD_new(float eta,
                                                        float momentum) {
  return to_c_ptr(new MomentumSGD(eta, momentum));
}

void primitiv_optimizers_MomentumSGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_MOMENTUM_SGD(optimizer);
}

float primitiv_optimizers_MomentumSGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->eta();
}

float primitiv_optimizers_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->momentum();
}

primitiv_Optimizer *primitiv_optimizers_AdaGrad_new(float eta, float eps) {
  return to_c_ptr(new AdaGrad(eta, eps));
}

void primitiv_optimizers_AdaGrad_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADA_GRAD(optimizer);
}

float primitiv_optimizers_AdaGrad_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
}

float primitiv_optimizers_AdaGrad_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eps();
}

primitiv_Optimizer *primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps) {
  return to_c_ptr(new RMSProp(eta, alpha, eps));
}

void primitiv_optimizers_RMSProp_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_RMS_PROP(optimizer);
}

float primitiv_optimizers_RMSProp_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
}

float primitiv_optimizers_RMSProp_alpha(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_RMS_PROP(optimizer)->alpha();
}

float primitiv_optimizers_RMSProp_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_RMS_PROP(optimizer)->eps();
}

primitiv_Optimizer *primitiv_optimizers_AdaDelta_new(float rho, float eps) {
  return to_c_ptr(new AdaDelta(rho, eps));
}

void primitiv_optimizers_AdaDelta_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADA_DELTA(optimizer);
}

float primitiv_optimizers_AdaDelta_rho(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_DELTA(optimizer)->rho();
}

float primitiv_optimizers_AdaDelta_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADA_DELTA(optimizer)->eps();
}

primitiv_Optimizer *primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps) {
  return to_c_ptr(new Adam(alpha, beta1, beta2, eps));
}

void primitiv_optimizers_Adam_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_ADAM(optimizer);
}

float primitiv_optimizers_Adam_alpha(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->alpha();
}

float primitiv_optimizers_Adam_beta1(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->beta1();
}

float primitiv_optimizers_Adam_beta2(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->beta2();
}

float primitiv_optimizers_Adam_eps(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_ADAM(optimizer)->eps();
}

}  // end extern "C"

#undef CAST_TO_CC_SGD
#undef CAST_TO_CONST_CC_SGD
#undef CAST_TO_CC_MOMENTUM_SGD
#undef CAST_TO_CONST_CC_MOMENTUM_SGD
#undef CAST_TO_CC_ADA_GRAD
#undef CAST_TO_CONST_CC_ADA_GRAD
#undef CAST_TO_CC_RMS_PROP
#undef CAST_TO_CONST_CC_RMS_PROP
#undef CAST_TO_CC_ADA_DELTA
#undef CAST_TO_CONST_CC_ADA_DELTA
#undef CAST_TO_CC_ADAM
#undef CAST_TO_CONST_CC_ADAM
