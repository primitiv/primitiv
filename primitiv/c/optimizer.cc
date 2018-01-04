#include <primitiv/config.h>

#include <primitiv/optimizer.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/optimizer.h>

using primitiv::Optimizer;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitiv_Optimizer_delete(primitiv_Optimizer *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  delete to_cpp_ptr(optimizer);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_load(
    primitiv_Optimizer *optimizer, const char *path) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(optimizer)->load(path);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_save(
    const primitiv_Optimizer *optimizer, const char *path) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(optimizer)->save(path);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_get_epoch(
    const primitiv_Optimizer *optimizer, uint32_t *epoch) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *epoch = to_cpp_ptr(optimizer)->get_epoch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_set_epoch(
    primitiv_Optimizer *optimizer, uint32_t epoch) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_epoch(epoch);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_get_learning_rate_scaling(
    const primitiv_Optimizer *optimizer, float *scale) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *scale = to_cpp_ptr(optimizer)->get_learning_rate_scaling();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_set_learning_rate_scaling(
    primitiv_Optimizer *optimizer, float scale) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_learning_rate_scaling(scale);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_get_weight_decay(
    const primitiv_Optimizer *optimizer, float *strength) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *strength = to_cpp_ptr(optimizer)->get_weight_decay();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_set_weight_decay(
    primitiv_Optimizer *optimizer, float strength) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_weight_decay(strength);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_get_gradient_clipping(
    const primitiv_Optimizer *optimizer, float *threshold) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  *threshold = to_cpp_ptr(optimizer)->get_gradient_clipping();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_set_gradient_clipping(
    primitiv_Optimizer *optimizer, float threshold) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_gradient_clipping(threshold);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_add_parameter(
    primitiv_Optimizer *optimizer, primitiv_Parameter *param) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(optimizer)->add(*to_cpp_ptr(param));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_add_parameters(
    primitiv_Optimizer *optimizer, primitiv_Parameter **params, size_t n) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cpp_ptr(params[i]));
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_add_model(
    primitiv_Optimizer *optimizer, primitiv_Model *model) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(model);
  to_cpp_ptr(optimizer)->add(*to_cpp_ptr(model));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_add_models(
    primitiv_Optimizer *optimizer, primitiv_Model **models, size_t n) try {
  Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cpp_ptr(models[i]));
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_reset_gradients(
    primitiv_Optimizer *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->reset_gradients();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Optimizer_update(primitiv_Optimizer *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->update();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
