#ifndef PRIMITIV_C_OPTIMIZER_IMPL_H_
#define PRIMITIV_C_OPTIMIZER_IMPL_H_

#include "primitiv_c/define.h"
#include "primitiv_c/optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Optimizer* primitiv_SGD_new();

primitiv_Optimizer* primitiv_SGD_new_with_eta(float eta);

void primitiv_SGD_delete(primitiv_Optimizer *optimizer);

float primitiv_SGD_eta(const primitiv_Optimizer *optimizer);

primitiv_Optimizer* primitiv_MomentumSGD_new();

primitiv_Optimizer* primitiv_MomentumSGD_new_with_eta(float eta);

primitiv_Optimizer* primitiv_MomentumSGD_new_with_eta_and_momentum(float eta, float momentum);

void primitiv_MomentumSGD_delete(primitiv_Optimizer *optimizer);

float primitiv_MomentumSGD_eta(const primitiv_Optimizer *optimizer);

float primitiv_MomentumSGD_momentum(const primitiv_Optimizer *optimizer);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPTIMIZER_IMPL_H_
