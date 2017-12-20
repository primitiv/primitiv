/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <vector>

#include <primitiv/parameter.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/parameter.h>

using primitiv::Parameter;

extern "C" {

primitiv_Parameter *primitiv_Parameter_new() {
  return to_c(new Parameter());
}
primitiv_Parameter *safe_primitiv_Parameter_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_new(), status, nullptr);
}

primitiv_Parameter *primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device) {
  return to_c(new Parameter(*to_cc(shape),
                            std::vector<float>(value, value + n),
                            *to_cc(device)));
}
primitiv_Parameter *safe_primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_Parameter_new_with_values(shape, value, n, device),
      status,
      nullptr);
}

primitiv_Parameter *primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device) {
  return to_c(new Parameter(*to_cc(shape),
                            *to_cc(initializer),
                            *to_cc(device)));
}
primitiv_Parameter *safe_primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_Parameter_new_with_initializer(shape, initializer, device),
      status,
      nullptr);
}

void primitiv_Parameter_delete(primitiv_Parameter *parameter) {
  delete to_cc(parameter);
}
void safe_primitiv_Parameter_delete(primitiv_Parameter *parameter,
                                    primitiv_Status *status) {
  SAFE_EXPR(primitiv_Parameter_delete(parameter), status);
}

void primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device) {
  to_cc(parameter)->init(*to_cc(shape),
                         std::vector<float>(value, value + n),
                         *to_cc(device));
}
void safe_primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Parameter_init_with_values(parameter, shape, value, n, device),
      status);
}

void primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device) {
  to_cc(parameter)->init(*to_cc(shape),
                         *to_cc(initializer),
                         *to_cc(device));
}
void safe_primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Parameter_init_with_initializer(
          parameter, shape, initializer, device), status);
}

void primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device) {
  to_cc(parameter)->load(path, with_stats, *to_cc(device));
}
void safe_primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device,
    primitiv_Status *status) {
  SAFE_EXPR(
      primitiv_Parameter_load(parameter, path, with_stats, device), status);
}

void primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats) {
  to_cc(parameter)->save(path, with_stats);
}
void safe_primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Status *status) {
  SAFE_EXPR(primitiv_Parameter_save(parameter, path, with_stats), status);
}

bool primitiv_Parameter_valid(const primitiv_Parameter *parameter) {
  return to_cc(parameter)->valid();
}
bool safe_primitiv_Parameter_valid(const primitiv_Parameter *parameter,
                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_valid(parameter), status, false);
}

void primitiv_Parameter_reset_gradients(primitiv_Parameter *parameter) {
  to_cc(parameter)->reset_gradient();
}
void safe_primitiv_Parameter_reset_gradients(primitiv_Parameter *parameter,
                                             primitiv_Status *status) {
  SAFE_EXPR(primitiv_Parameter_reset_gradients(parameter), status);
}

void primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape) {
  to_cc(parameter)->add_stats(name, *to_cc(shape));
}
void safe_primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape,
    primitiv_Status *status) {
  SAFE_EXPR(primitiv_Parameter_add_stats(parameter, name, shape), status);
}

bool primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name) {
  return to_cc(parameter)->has_stats(name);
}
bool safe_primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_has_stats(parameter, name), status, false);
}

primitiv_Shape *primitiv_Parameter_shape(const primitiv_Parameter *parameter) {
  return to_c_from_value(to_cc(parameter)->shape());
}
primitiv_Shape *safe_primitiv_Parameter_shape(
    const primitiv_Parameter *parameter, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_shape(parameter), status, nullptr);
}

primitiv_Device *primitiv_Parameter_device(
    const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->device());
}
primitiv_Device *safe_primitiv_Parameter_device(
    const primitiv_Parameter *parameter, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_device(parameter), status, nullptr);
}

const primitiv_Tensor *primitiv_Parameter_value(
    const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->value());
}
const primitiv_Tensor *safe_primitiv_Parameter_value(
    const primitiv_Parameter *parameter, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_value(parameter), status, nullptr);
}

const primitiv_Tensor *primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->gradient());
}
const primitiv_Tensor *safe_primitiv_Parameter_gradient(
    const primitiv_Parameter *parameter, primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_gradient(parameter), status, nullptr);
}

const primitiv_Tensor *primitiv_Parameter_stats(
    const primitiv_Parameter *parameter, const char *name) {
  return to_c(&to_cc(parameter)->stats(name));
}
const primitiv_Tensor *safe_primitiv_Parameter_stats(
    const primitiv_Parameter *parameter,
    const char *name,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Parameter_stats(parameter, name), status, nullptr);
}

}  // end extern "C"
