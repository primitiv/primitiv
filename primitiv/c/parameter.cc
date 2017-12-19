/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <vector>

#include <primitiv/parameter.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;
using primitiv_c::internal::to_c;
using primitiv_c::internal::to_cc;
using primitiv_c::internal::to_c_from_value;

extern "C" {

primitiv_Parameter *primitiv_Parameter_new() {
  return to_c(new Parameter());
}

primitiv_Status primitiv_Parameter_new_with_values(
    primitiv_Parameter **parameter, const primitiv_Shape *shape,
    const float *value, size_t n, primitiv_Device *device) {
  try {
    *parameter = to_c(new Parameter(
        *to_cc(shape), std::vector<float>(value, value + n), *to_cc(device)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_new_with_initializer(
    primitiv_Parameter **parameter, const primitiv_Shape *shape,
    const primitiv_Initializer *initializer, primitiv_Device *device) {
  try {
    *parameter = to_c(new Parameter(
        *to_cc(shape), *to_cc(initializer), *to_cc(device)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_Parameter_delete(primitiv_Parameter *parameter) {
  delete to_cc(parameter);
}

primitiv_Status primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const float *value, size_t n, primitiv_Device *device) {
  try {
    to_cc(parameter)->init(
        *to_cc(shape), std::vector<float>(value, value + n), *to_cc(device));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter, const primitiv_Shape *shape,
    const primitiv_Initializer *initializer, primitiv_Device *device) {
  try {
    to_cc(parameter)->init(*to_cc(shape), *to_cc(initializer), *to_cc(device));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device) {
  try {
    to_cc(parameter)->load(path, with_stats, *to_cc(device));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats) {
  try {
    to_cc(parameter)->save(path, with_stats);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

bool primitiv_Parameter_valid(const primitiv_Parameter *parameter) {
  return to_cc(parameter)->valid();
}

primitiv_Status primitiv_Parameter_reset_gradients(
    primitiv_Parameter *parameter) {
  try {
    to_cc(parameter)->reset_gradient();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape) {
  try {
    to_cc(parameter)->add_stats(name, *to_cc(shape));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter, const char *name, bool *has_stats) {
  try {
    *has_stats = to_cc(parameter)->has_stats(name);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_shape(const primitiv_Parameter *parameter,
                                         primitiv_Shape **shape) {
  try {
    *shape = to_c_from_value(to_cc(parameter)->shape());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_device(const primitiv_Parameter *parameter,
                                          primitiv_Device **device) {
  try {
    *device = to_c(&to_cc(parameter)->device());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_value(const primitiv_Parameter *parameter,
                                         const primitiv_Tensor **tensor) {
  try {
    *tensor = to_c(&to_cc(parameter)->value());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_gradient(const primitiv_Parameter *parameter,
                                            const primitiv_Tensor **tensor) {
  try {
    *tensor = to_c(&to_cc(parameter)->gradient());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Parameter_stats(const primitiv_Parameter *parameter,
                                         const char *name,
                                         const primitiv_Tensor **tensor) {
  try {
    *tensor = to_c(&to_cc(parameter)->stats(name));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
