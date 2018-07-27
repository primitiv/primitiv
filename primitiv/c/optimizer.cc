#include <primitiv/config.h>

#include <string>
#include <unordered_map>

#include <primitiv/core/optimizer.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/optimizer.h>

using primitiv::Optimizer;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitivDeleteOptimizer(primitivOptimizer_t *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  delete to_cpp_ptr(optimizer);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivLoadOptimizer(
    primitivOptimizer_t *optimizer, const char *path) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(optimizer)->load(path);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSaveOptimizer(
    const primitivOptimizer_t *optimizer, const char *path) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(optimizer)->save(path);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerEpoch(
    const primitivOptimizer_t *optimizer, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(optimizer)->get_epoch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerEpoch(
    primitivOptimizer_t *optimizer, uint32_t epoch) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_epoch(epoch);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerLearningRateScaling(
    const primitivOptimizer_t *optimizer, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(optimizer)->get_learning_rate_scaling();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerLearningRateScaling(
    primitivOptimizer_t *optimizer, float scale) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_learning_rate_scaling(scale);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerWeightDecay(
    const primitivOptimizer_t *optimizer, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(optimizer)->get_weight_decay();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerWeightDecay(
    primitivOptimizer_t *optimizer, float strength) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_weight_decay(strength);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerGradientClipping(
    const primitivOptimizer_t *optimizer, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(optimizer)->get_gradient_clipping();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerGradientClipping(
    primitivOptimizer_t *optimizer, float threshold) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->set_gradient_clipping(threshold);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddParameterToOptimizer(
    primitivOptimizer_t *optimizer, primitivParameter_t *param) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(optimizer)->add(*to_cpp_ptr(param));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddParametersToOptimizer(
    primitivOptimizer_t *optimizer, primitivParameter_t **params,
    size_t n) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(params);
  Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cpp_ptr(params[i]));
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddModelToOptimizer(
    primitivOptimizer_t *optimizer, primitivModel_t *model) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(model);
  to_cpp_ptr(optimizer)->add(*to_cpp_ptr(model));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddModelsToOptimizer(
    primitivOptimizer_t *optimizer, primitivModel_t **models, size_t n) try {
  Optimizer *cc_optimizer = to_cpp_ptr(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(models);
  for (size_t i = 0; i < n; ++i) {
    cc_optimizer->add(*to_cpp_ptr(models[i]));
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResetOptimizerGradients(
    primitivOptimizer_t *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->reset_gradients();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivExecuteOptimizerUpdate(
    primitivOptimizer_t *optimizer) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  to_cpp_ptr(optimizer)->update();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerIntConfig(
    const primitivOptimizer_t *optimizer, const char *key,
    uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(key);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  std::unordered_map<std::string, uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  to_cpp_ptr(optimizer)->get_configs(uint_configs, float_configs);
  const auto it = uint_configs.find(key);
  if (it != uint_configs.end()) {
    *retval = uint_configs[key];
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerIntConfig(
    primitivOptimizer_t *optimizer, const char *key,
    uint32_t value) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(key);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  std::unordered_map<std::string, uint32_t> uint_configs{{key, value}};
  std::unordered_map<std::string, float> float_configs;
  to_cpp_ptr(optimizer)->set_configs(uint_configs, float_configs);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetOptimizerFloatConfig(
    const primitivOptimizer_t *optimizer, const char *key, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(key);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  std::unordered_map<std::string, uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  to_cpp_ptr(optimizer)->get_configs(uint_configs, float_configs);
  const auto it = float_configs.find(key);
  if (it != float_configs.end()) {
    *retval = float_configs[key];
  }
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSetOptimizerFloatConfig(
    primitivOptimizer_t *optimizer, const char *key, float value) try {
  PRIMITIV_C_CHECK_NOT_NULL(optimizer);
  PRIMITIV_C_CHECK_NOT_NULL(key);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  std::unordered_map<std::string, uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs{{key, value}};
  to_cpp_ptr(optimizer)->set_configs(uint_configs, float_configs);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
