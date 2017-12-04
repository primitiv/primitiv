#include "primitiv_c/internal.h"
#include "primitiv_c/optimizer.h"

#include <primitiv/optimizer.h>

using primitiv::Optimizer;

extern "C" {

primitiv_Optimizer *primitiv_Optimizer_new() {
  // throw exception
  return nullptr;
}

void primitiv_Optimizer_delete(primitiv_Optimizer *optimizer) {
  // throw exception
}

void primitiv_Optimizer_load(primitiv_Optimizer *optimizer, const char *path) {
  to_cc(optimizer)->load(path);
}

void primitiv_Optimizer_save(const primitiv_Optimizer *optimizer, const char *path) {
  to_cc(optimizer)->save(path);
}

uint32_t primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_epoch();
}

void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer, uint32_t epoch) {
  to_cc(optimizer)->set_epoch(epoch);
}

float primitiv_Optimizer_get_learning_rate_scaling(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_learning_rate_scaling();
}

void primitiv_Optimizer_set_learning_rate_scaling(primitiv_Optimizer *optimizer, float scale) {
  to_cc(optimizer)->set_epoch(scale);
}

float primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_weight_decay();
}

void primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer, float strength) {
  to_cc(optimizer)->set_weight_decay(strength);
}

float primitiv_Optimizer_get_gradient_clipping(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_gradient_clipping();
}

void primitiv_Optimizer_set_gradient_clipping(primitiv_Optimizer *optimizer, float threshold) {
  to_cc(optimizer)->set_gradient_clipping(threshold);
}

void primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer, primitiv_Parameter *param) {
  to_cc(optimizer)->add(*to_cc(param));
}

void primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer, const primitiv_Model *model) {
  to_cc(optimizer)->add(*to_cc(model));
}

void primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer) {
  to_cc(optimizer)->reset_gradients();
}

void primitiv_Optimizer_update(primitiv_Optimizer *optimizer) {
  to_cc(optimizer)->update();
}

}  // end extern "C"
