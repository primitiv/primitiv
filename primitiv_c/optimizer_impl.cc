#include "primitiv_c/internal.h"
#include "primitiv_c/optimizer_impl.h"

#include <primitiv/optimizer_impl.h>

using primitiv::optimizers::SGD;
using primitiv::optimizers::MomentumSGD;

#define CAST_TO_CC_SGD(x) reinterpret_cast<SGD*>(x)
#define CAST_TO_CONST_CC_SGD(x) reinterpret_cast<const SGD*>(x)

#define CAST_TO_CC_MOMENTUM_SGD(x) reinterpret_cast<MomentumSGD*>(x)
#define CAST_TO_CONST_CC_MOMENTUM_SGD(x) reinterpret_cast<const MomentumSGD*>(x)

extern "C" {

primitiv_Optimizer *primitiv_SGD_new() {
  return to_c(new SGD());
}

primitiv_Optimizer *primitiv_SGD_new_with_eta(float eta) {
  return to_c(new SGD(eta));
}

void primitiv_SGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_SGD(optimizer);
}

float primitiv_SGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_SGD(optimizer)->eta();
}

primitiv_Optimizer *primitiv_MomentumSGD_new() {
  return to_c(new MomentumSGD());
}

primitiv_Optimizer *primitiv_MomentumSGD_new_with_eta(float eta) {
  return to_c(new MomentumSGD(eta));
}

primitiv_Optimizer *primitiv_MomentumSGD_new_with_eta_and_momentum(float eta, float momentum) {
  return to_c(new MomentumSGD(eta, momentum));
}

void primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer) {
  delete CAST_TO_CC_MOMENTUM_SGD(optimizer);
}

float primitiv_MomentumSGD_eta(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->eta();
}

float primitiv_MomentumSGD_momentum(const primitiv_Optimizer *optimizer) {
  return CAST_TO_CONST_CC_MOMENTUM_SGD(optimizer)->momentum();
}

}  // end extern "C"
