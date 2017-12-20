/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <algorithm>
#include <vector>

#include <primitiv/tensor.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/tensor.h>

using primitiv::Tensor;

extern "C" {

primitiv_Tensor *primitiv_Tensor_new() {
  return to_c(new Tensor());
}
primitiv_Tensor *safe_primitiv_Tensor_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_new(), status, nullptr);
}

primitiv_Tensor *primitiv_Tensor_new_from_tensor(primitiv_Tensor *tensor) {
  return to_c(new Tensor(*to_cc(tensor)));
}
primitiv_Tensor *safe_primitiv_Tensor_new_from_tensor(primitiv_Tensor *tensor,
                                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_new_from_tensor(tensor), status, nullptr);
}

void primitiv_Tensor_delete(primitiv_Tensor *tensor) {
  delete to_cc(tensor);
}
void safe_primitiv_Tensor_delete(primitiv_Tensor *tensor,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_Tensor_delete(tensor), status);
}

bool primitiv_Tensor_valid(const primitiv_Tensor *tensor) {
  return to_cc(tensor)->valid();
}
bool safe_primitiv_Tensor_valid(const primitiv_Tensor *tensor,
                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_valid(tensor), status, false);
}

primitiv_Shape *primitiv_Tensor_shape(const primitiv_Tensor *tensor) {
  return to_c_from_value(to_cc(tensor)->shape());
}
primitiv_Shape *safe_primitiv_Tensor_shape(const primitiv_Tensor *tensor,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_shape(tensor), status, nullptr);
}

primitiv_Device *primitiv_Tensor_device(const primitiv_Tensor *tensor) {
  return to_c(&to_cc(tensor)->device());
}
primitiv_Device *safe_primitiv_Tensor_device(const primitiv_Tensor *tensor,
                                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_device(tensor), status, nullptr);
}

float primitiv_Tensor_to_float(const primitiv_Tensor *tensor) {
  return to_cc(tensor)->to_float();
}
float safe_primitiv_Tensor_to_float(const primitiv_Tensor *tensor,
                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_to_float(tensor), status, 0.0);
}

void primitiv_Tensor_to_array(const primitiv_Tensor *tensor, float *array) {
  std::vector<float> v = to_cc(tensor)->to_vector();
  std::copy(v.begin(), v.end(), array);
}
void safe_primitiv_Tensor_to_array(const primitiv_Tensor *tensor,
                                   float *array,
                                   primitiv_Status *status) {
  SAFE_EXPR(primitiv_Tensor_to_array(tensor, array), status);
}

uint32_t *primitiv_Tensor_argmax(const primitiv_Tensor *tensor, uint32_t dim) {
  return &(to_cc(tensor)->argmax(dim))[0];
}
uint32_t *safe_primitiv_Tensor_argmax(const primitiv_Tensor *tensor,
                                      uint32_t dim,
                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_argmax(tensor, dim), status, nullptr);
}

uint32_t *primitiv_Tensor_argmin(const primitiv_Tensor *tensor, uint32_t dim) {
  return &(to_cc(tensor)->argmin(dim))[0];
}
uint32_t *safe_primitiv_Tensor_argmin(const primitiv_Tensor *tensor,
                                      uint32_t dim,
                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_argmin(tensor, dim), status, nullptr);
}

void primitiv_Tensor_reset(primitiv_Tensor *tensor, float k) {
  to_cc(tensor)->reset(k);
}
void safe_primitiv_Tensor_reset(primitiv_Tensor *tensor,
                                float k,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Tensor_reset(tensor, k), status);
}

void primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                    const float *values) {
  to_cc(tensor)->reset_by_array(values);
}
void safe_primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                         const float *values,
                                         primitiv_Status *status) {
  SAFE_EXPR(primitiv_Tensor_reset_by_array(tensor, values), status);
}

primitiv_Tensor *primitiv_Tensor_reshape(const primitiv_Tensor *tensor,
                                         const primitiv_Shape *new_shape) {
  return to_c_from_value(to_cc(tensor)->reshape(*to_cc(new_shape)));
}
primitiv_Tensor *safe_primitiv_Tensor_reshape(const primitiv_Tensor *tensor,
                                              const primitiv_Shape *new_shape,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_reshape(tensor, new_shape), status, nullptr);
}

primitiv_Tensor *primitiv_Tensor_flatten(const primitiv_Tensor *tensor) {
  return to_c_from_value(to_cc(tensor)->flatten());
}
primitiv_Tensor *safe_primitiv_Tensor_flatten(const primitiv_Tensor *tensor,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_flatten(tensor), status, nullptr);
}

primitiv_Tensor *primitiv_Tensor_inplace_multiply_const(primitiv_Tensor *tensor,
                                                        float k) {
  return to_c(&(to_cc(tensor)->inplace_multiply_const(k)));
}
primitiv_Tensor *safe_primitiv_Tensor_inplace_multiply_const(
    primitiv_Tensor *tensor, float k, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_Tensor_inplace_multiply_const(tensor, k), status, nullptr);
}

primitiv_Tensor *primitiv_Tensor_inplace_add(primitiv_Tensor *tensor,
                                             const primitiv_Tensor *x) {
  return to_c(&(to_cc(tensor)->inplace_add(*to_cc(x))));
}
primitiv_Tensor *safe_primitiv_Tensor_inplace_add(primitiv_Tensor *tensor,
                                                  const primitiv_Tensor *x,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_inplace_add(tensor, x), status, nullptr);
}

primitiv_Tensor *primitiv_Tensor_inplace_subtract(primitiv_Tensor *tensor,
                                                  const primitiv_Tensor *x) {
  return to_c(&(to_cc(tensor)->inplace_subtract(*to_cc(x))));
}
primitiv_Tensor *safe_primitiv_Tensor_inplace_subtract(
    primitiv_Tensor *tensor,
    const primitiv_Tensor *x,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Tensor_inplace_subtract(tensor, x), status, nullptr);
}

}  // end extern "C"
