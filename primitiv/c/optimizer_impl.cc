#include <primitiv/config.h>

#include <primitiv/core/optimizer_impl.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/optimizer_impl.h>

using primitiv::optimizers::SGD;
using primitiv::optimizers::MomentumSGD;
using primitiv::optimizers::AdaGrad;
using primitiv::optimizers::RMSProp;
using primitiv::optimizers::AdaDelta;
using primitiv::optimizers::Adam;
using primitiv::c::internal::to_c_ptr;

PRIMITIV_C_STATUS primitivCreateSgdOptimizer(
    float eta, primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new SGD(eta));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateMomentumSgdOptimizer(
    float eta, float momentum, primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new MomentumSGD(eta, momentum));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateAdaGradOptimizer(
    float eta, float eps, primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new AdaGrad(eta, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateRmsPropOptimizer(
    float eta, float alpha, float eps, primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new RMSProp(eta, alpha, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateAdaDeltaOptimizer(
    float rho, float eps, primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new AdaDelta(rho, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateAdamOptimizer(
    float alpha, float beta1, float beta2, float eps,
    primitivOptimizer_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Adam(alpha, beta1, beta2, eps));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
