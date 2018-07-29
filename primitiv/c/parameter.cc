#include <primitiv/config.h>

#include <vector>

#include <primitiv/core/parameter.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitivCreateParameter(primitivParameter_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Parameter());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateParameterWithValues(
    const primitivShape_t *shape, const float *value, size_t n,
    primitivDevice_t *device, primitivParameter_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Parameter(
        *to_cpp_ptr(shape),
        std::vector<float>(value, value + n),
        *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateParameterWithInitializer(
    const primitivShape_t *shape, const primitivInitializer_t *initializer,
    primitivDevice_t *device, primitivParameter_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Parameter(
      *to_cpp_ptr(shape), *to_cpp_ptr(initializer), *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteParameter(primitivParameter_t *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  delete to_cpp_ptr(parameter);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivInitializeParameterWithValues(
    primitivParameter_t *parameter, const primitivShape_t *shape,
    const float *value, size_t n, primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  to_cpp_ptr(parameter)->init(
      *to_cpp_ptr(shape),
      std::vector<float>(value, value + n),
      *to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivInitializeParameterWithInitializer(
    primitivParameter_t *parameter, const primitivShape_t *shape,
    const primitivInitializer_t *initializer, primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  to_cpp_ptr(parameter)->init(
      *to_cpp_ptr(shape),
      *to_cpp_ptr(initializer),
      *to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivLoadParameter(
    primitivParameter_t *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats,
    primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->load(path, with_stats, *to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSaveParameter(
    const primitivParameter_t *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->save(path, with_stats);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsValidParameter(
    const primitivParameter_t *parameter, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(parameter)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResetParameterGradients(
    primitivParameter_t *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  to_cpp_ptr(parameter)->reset_gradient();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddStatsToParameter(
    primitivParameter_t *parameter,
    const char *name,
    const primitivShape_t *shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(parameter)->add_stats(name, *to_cpp_ptr(shape));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivHasParameterStats(
    const primitivParameter_t *parameter, const char *name,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(parameter)->has_stats(name);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetParameterShape(
    const primitivParameter_t *parameter,
    primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(parameter)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromParameter(
    const primitivParameter_t *parameter,
    primitivDevice_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(parameter)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetParameterValue(
    const primitivParameter_t *parameter,
    const primitivTensor_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(parameter)->value());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetParameterGradient(
    const primitivParameter_t *parameter,
    const primitivTensor_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(parameter)->gradient());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetParameterStats(
    const primitivParameter_t *parameter,
    const char *name,
    const primitivTensor_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(parameter)->stats(name));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
