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

PRIMITIV_C_STATUS primitiv_optimizers_SGD_new(
    float eta, primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new SGD(eta));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_optimizers_MomentumSGD_new(
    float eta, float momentum, primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new MomentumSGD(eta, momentum));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_optimizers_AdaGrad_new(
    float eta, float eps, primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new AdaGrad(eta, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_optimizers_RMSProp_new(
    float eta, float alpha, float eps, primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new RMSProp(eta, alpha, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_optimizers_AdaDelta_new(
    float rho, float eps, primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new AdaDelta(rho, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_optimizers_Adam_new(
    float alpha, float beta1, float beta2, float eps,
    primitiv_Optimizer **optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *optimizer = to_c_ptr(new Adam(alpha, beta1, beta2, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
