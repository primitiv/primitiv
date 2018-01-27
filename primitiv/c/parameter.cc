#include <primitiv/config.h>

#include <vector>

#include <primitiv/parameter.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitiv_Parameter_new(primitivParameter_t **parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *parameter = to_c_ptr(new Parameter());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_new_with_values(
    const primitivShape_t *shape, const float *value, size_t n,
    primitivDevice_t *device, primitivParameter_t **parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *parameter = to_c_ptr(new Parameter(
        *to_cpp_ptr(shape),
        std::vector<float>(value, value + n),
        *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_new_with_initializer(
    const primitivShape_t *shape, const primitivInitializer_t *initializer,
    primitivDevice_t *device, primitivParameter_t **parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *parameter = to_c_ptr(new Parameter(
      *to_cpp_ptr(shape), *to_cpp_ptr(initializer), *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_delete(primitivParameter_t *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  delete to_cpp_ptr(parameter);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_init_with_values(
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

PRIMITIV_C_STATUS primitiv_Parameter_init_with_initializer(
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

PRIMITIV_C_STATUS primitiv_Parameter_load(
    primitivParameter_t *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats,
    primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->load(path, with_stats, *to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_save(
    const primitivParameter_t *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->save(path, with_stats);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_valid(
    const primitivParameter_t *parameter, PRIMITIV_C_BOOL *valid) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(valid);
  *valid = to_cpp_ptr(parameter)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_reset_gradients(
    primitivParameter_t *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  to_cpp_ptr(parameter)->reset_gradient();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_add_stats(
    primitivParameter_t *parameter,
    const char *name,
    const primitivShape_t *shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(parameter)->add_stats(name, *to_cpp_ptr(shape));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_has_stats(
    primitivParameter_t *parameter, const char *name,
    PRIMITIV_C_BOOL *has_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(has_stats);
  *has_stats = to_cpp_ptr(parameter)->has_stats(name);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_shape(
    const primitivParameter_t *parameter,
    primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(to_cpp_ptr(parameter)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_device(
    const primitivParameter_t *parameter,
    primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(parameter)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_value(
    const primitivParameter_t *parameter,
    const primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->value());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_gradient(
    const primitivParameter_t *parameter,
    const primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->gradient());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_stats(
    const primitivParameter_t *parameter,
    const char *name,
    const primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->stats(name));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
