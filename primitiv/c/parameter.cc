#include <primitiv/config.h>

#include <vector>

#include <primitiv/parameter.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitiv_Parameter_new(primitiv_Parameter **parameter) try {
  *parameter = to_c_ptr(new Parameter());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape, const float *value, size_t n,
    primitiv_Device *device, primitiv_Parameter **parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  *parameter = to_c_ptr(new Parameter(
        *to_cpp_ptr(shape),
        std::vector<float>(value, value + n),
        *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape, const primitiv_Initializer *initializer,
    primitiv_Device *device, primitiv_Parameter **parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(initializer);
  *parameter = to_c_ptr(new Parameter(
      *to_cpp_ptr(shape), *to_cpp_ptr(initializer), *to_cpp_ptr(device)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_delete(primitiv_Parameter *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  delete to_cpp_ptr(parameter);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const float *value, size_t n, primitiv_Device *device) try {
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
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const primitiv_Initializer *initializer, primitiv_Device *device) try {
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
    primitiv_Parameter *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats,
    primitiv_Device *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->load(path, with_stats, *to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    PRIMITIV_C_BOOL with_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(parameter)->save(path, with_stats);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_valid(
    const primitiv_Parameter *parameter, PRIMITIV_C_BOOL *valid) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *valid = to_cpp_ptr(parameter)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_reset_gradients(
    primitiv_Parameter *parameter) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  to_cpp_ptr(parameter)->reset_gradient();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(parameter)->add_stats(name, *to_cpp_ptr(shape));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter, const char *name,
    PRIMITIV_C_BOOL *has_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  *has_stats = to_cpp_ptr(parameter)->has_stats(name);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_shape(
    const primitiv_Parameter *parameter,
    primitiv_Shape **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *shape = to_c_ptr_from_value(to_cpp_ptr(parameter)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_device(
    const primitiv_Parameter *parameter,
    primitiv_Device **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *device = to_c_ptr(&to_cpp_ptr(parameter)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_value(
    const primitiv_Parameter *parameter,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->value());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->gradient());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Parameter_stats(
    const primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(parameter);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->stats(name));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
