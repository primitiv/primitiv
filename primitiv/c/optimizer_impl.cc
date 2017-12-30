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

primitiv_Status primitiv_optimizers_SGD_new(
    float eta, primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new SGD(eta));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_SGD_eta(
    const primitiv_Optimizer *optimizer, float *eta) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eta = CAST_TO_CONST_CC_SGD(optimizer)->eta();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_MomentumSGD_new(
    float eta, float momentum, primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new MomentumSGD(eta, momentum));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_MomentumSGD_eta(
    const primitiv_Optimizer *optimizer, float *eta) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eta = CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->eta();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_MomentumSGD_momentum(
    const primitiv_Optimizer *optimizer, float *momentum) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *momentum = CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->momentum();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaGrad_new(
    float eta, float eps, primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new AdaGrad(eta, eps));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaGrad_eta(
    const primitiv_Optimizer *optimizer, float *eta) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eta = CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaGrad_eps(
    const primitiv_Optimizer *optimizer, float *eps) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eps = CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eps();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps, primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new RMSProp(eta, alpha, eps));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_RMSProp_eta(
    const primitiv_Optimizer *optimizer, float *eta) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eta = CAST_TO_CONST_CC_ADA_GRAD(optimizer)->eta();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_RMSProp_alpha(
    const primitiv_Optimizer *optimizer, float *alpha) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *alpha = CAST_TO_CONST_CC_RMS_PROP(optimizer)->alpha();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_RMSProp_eps(
    const primitiv_Optimizer *optimizer, float *eps) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eps = CAST_TO_CONST_CC_RMS_PROP(optimizer)->eps();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaDelta_new(
    float rho, float eps, primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new AdaDelta(rho, eps));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaDelta_rho(
    const primitiv_Optimizer *optimizer, float *rho) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *rho = CAST_TO_CONST_CC_ADA_DELTA(optimizer)->rho();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_AdaDelta_eps(
    const primitiv_Optimizer *optimizer, float *eps) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eps = CAST_TO_CONST_CC_ADA_DELTA(optimizer)->eps();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps,
    primitiv_Optimizer **optimizer) try {
  *optimizer = to_c_ptr(new Adam(alpha, beta1, beta2, eps));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_Adam_alpha(
    const primitiv_Optimizer *optimizer, float *alpha) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *alpha = CAST_TO_CONST_CC_ADAM(optimizer)->alpha();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_Adam_beta1(
    const primitiv_Optimizer *optimizer, float *beta1) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *beta1 = CAST_TO_CONST_CC_ADAM(optimizer)->beta1();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_Adam_beta2(
    const primitiv_Optimizer *optimizer, float *beta2) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *beta2 = CAST_TO_CONST_CC_ADAM(optimizer)->beta2();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_optimizers_Adam_eps(
    const primitiv_Optimizer *optimizer, float *eps) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *eps = CAST_TO_CONST_CC_ADAM(optimizer)->eps();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

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
