/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <vector>

#include <primitiv/parameter.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

extern "C" {

primitiv_Status primitiv_Parameter_new(primitiv_Parameter **parameter) try {
  *parameter = to_c_ptr(new Parameter());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape, const float *value, size_t n,
    primitiv_Device *device, primitiv_Parameter **parameter) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(value);
  *parameter = to_c_ptr(new Parameter(
        *to_cpp_ptr(shape),
        std::vector<float>(value, value + n),
        *to_cpp_ptr(device)));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape, const primitiv_Initializer *initializer,
    primitiv_Device *device, primitiv_Parameter **parameter) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(initializer);
  *parameter = to_c_ptr(new Parameter(
      *to_cpp_ptr(shape), *to_cpp_ptr(initializer), *to_cpp_ptr(device)));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_delete(primitiv_Parameter *parameter) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  delete to_cpp_ptr(parameter);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const float *value, size_t n, primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(value);
  to_cpp_ptr(parameter)->init(
      *to_cpp_ptr(shape),
      std::vector<float>(value, value + n),
      *to_cpp_ptr(device));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const primitiv_Initializer *initializer, primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(initializer);
  to_cpp_ptr(parameter)->init(
      *to_cpp_ptr(shape),
      *to_cpp_ptr(initializer),
      *to_cpp_ptr(device));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    _Bool with_stats,
    primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(path);
  to_cpp_ptr(parameter)->load(path, with_stats, *to_cpp_ptr(device));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    _Bool with_stats) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(path);
  to_cpp_ptr(parameter)->save(path, with_stats);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_valid(
    const primitiv_Parameter *parameter, _Bool *valid) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  *valid = to_cpp_ptr(parameter)->valid();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_reset_gradients(
    primitiv_Parameter *parameter) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  to_cpp_ptr(parameter)->reset_gradient();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(name);
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  to_cpp_ptr(parameter)->add_stats(name, *to_cpp_ptr(shape));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter, const char *name, _Bool *has_stats) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(name);
  *has_stats = to_cpp_ptr(parameter)->has_stats(name);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_shape(
    const primitiv_Parameter *parameter,
    primitiv_Shape **shape) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  *shape = to_c_ptr_from_value(to_cpp_ptr(parameter)->shape());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_device(
    const primitiv_Parameter *parameter,
    primitiv_Device **device) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  *device = to_c_ptr(&to_cpp_ptr(parameter)->device());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_value(
    const primitiv_Parameter *parameter,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->value());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->gradient());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Parameter_stats(
    const primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Tensor **tensor) try {
  PRIMITIV_C_CHECK_PTR_ARG(parameter);
  PRIMITIV_C_CHECK_PTR_ARG(name);
  *tensor = to_c_ptr(&to_cpp_ptr(parameter)->stats(name));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
