/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/optimizer.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/optimizer.h>

using primitiv::Optimizer;

extern "C" {

void primitiv_Optimizer_load(primitiv_Optimizer *optimizer, const char *path) {
  to_cc(optimizer)->load(path);
}
void safe_primitiv_Optimizer_load(primitiv_Optimizer *optimizer,
                                  const char *path,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_load(optimizer, path), status);
}

void primitiv_Optimizer_save(const primitiv_Optimizer *optimizer,
                             const char *path) {
  to_cc(optimizer)->save(path);
}
void safe_primitiv_Optimizer_save(const primitiv_Optimizer *optimizer,
                                  const char *path,
                                  primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_save(optimizer, path), status);
}

uint32_t primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_epoch();
}
uint32_t safe_primitiv_Optimizer_get_epoch(const primitiv_Optimizer *optimizer,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Optimizer_get_epoch(optimizer), status, 0);
}

void primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer,
                                  uint32_t epoch) {
  to_cc(optimizer)->set_epoch(epoch);
}
void safe_primitiv_Optimizer_set_epoch(primitiv_Optimizer *optimizer,
                                       uint32_t epoch,
                                       primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_set_epoch(optimizer, epoch), status);
}

float primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_learning_rate_scaling();
}
float safe_primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_Optimizer_get_learning_rate_scaling(optimizer), status, 0.0);
}

void primitiv_Optimizer_set_learning_rate_scaling(primitiv_Optimizer *optimizer,
                                                  float scale) {
  to_cc(optimizer)->set_epoch(scale);
}
void safe_primitiv_Optimizer_set_learning_rate_scaling(
    primitiv_Optimizer *optimizer, float scale, primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Optimizer_set_learning_rate_scaling(optimizer, scale), status);
}

float primitiv_Optimizer_get_weight_decay(const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_weight_decay();
}
float safe_primitiv_Optimizer_get_weight_decay(
    const primitiv_Optimizer *optimizer, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Optimizer_get_weight_decay(optimizer), status, 0.0);
}

void primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer,
                                         float strength) {
  to_cc(optimizer)->set_weight_decay(strength);
}
void safe_primitiv_Optimizer_set_weight_decay(primitiv_Optimizer *optimizer,
                                              float strength,
                                              primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_set_weight_decay(optimizer, strength), status);
}

float primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer) {
  return to_cc(optimizer)->get_gradient_clipping();
}
float safe_primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Optimizer_get_gradient_clipping(optimizer), status, 0.0);
}

void primitiv_Optimizer_set_gradient_clipping(primitiv_Optimizer *optimizer,
                                              float threshold) {
  to_cc(optimizer)->set_gradient_clipping(threshold);
}
void safe_primitiv_Optimizer_set_gradient_clipping(
    primitiv_Optimizer *optimizer, float threshold, primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Optimizer_set_gradient_clipping(optimizer, threshold), status);
}

void primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer,
                                      primitiv_Parameter *param) {
  to_cc(optimizer)->add(*to_cc(param));
}
void safe_primitiv_Optimizer_add_parameter(primitiv_Optimizer *optimizer,
                                           primitiv_Parameter *param,
                                           primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_add_parameter(optimizer, param), status);
}

void primitiv_Optimizer_add_parameters(primitiv_Optimizer *optimizer,
                                       primitiv_Parameter **params, size_t n) {
  Optimizer *cc_optimizer = to_cc(optimizer);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cc(params[i]));
  }
}
void safe_primitiv_Optimizer_add_parameters(primitiv_Optimizer *optimizer,
                                            primitiv_Parameter **params,
                                            size_t n,
                                            primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_add_parameters(optimizer, params, n), status);
}

void primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer,
                                  primitiv_Model *model) {
  to_cc(optimizer)->add(*to_cc(model));
}
void safe_primitiv_Optimizer_add_model(primitiv_Optimizer *optimizer,
                                       primitiv_Model *model,
                                       primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_add_model(optimizer, model), status);
}

void primitiv_Optimizer_add_models(primitiv_Optimizer *optimizer,
                                   primitiv_Model **models,
                                   size_t n) {
  Optimizer *cc_optimizer = to_cc(optimizer);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cc(models[i]));
  }
}
void safe_primitiv_Optimizer_add_models(primitiv_Optimizer *optimizer,
                                        primitiv_Model **models,
                                        size_t n,
                                        primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_add_models(optimizer, models, n), status);
}

void primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer) {
  to_cc(optimizer)->reset_gradients();
}
void safe_primitiv_Optimizer_reset_gradients(primitiv_Optimizer *optimizer,
                                             primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_reset_gradients(optimizer), status);
}

void primitiv_Optimizer_update(primitiv_Optimizer *optimizer) {
  to_cc(optimizer)->update();
}
void safe_primitiv_Optimizer_update(primitiv_Optimizer *optimizer,
                                    primitiv_Status *status) {
  SAFE_EXPR(primitiv_Optimizer_update(optimizer), status);
}

}  // end extern "C"
