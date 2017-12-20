/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <algorithm>
#include <vector>

#include <primitiv/tensor.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/tensor.h>

using primitiv::Tensor;
using primitiv::c::internal::to_c;
using primitiv::c::internal::to_cc;
using primitiv::c::internal::to_c_from_value;

extern "C" {

primitiv_Tensor *primitiv_Tensor_new() {
  return to_c(new Tensor());
}

primitiv_Tensor *primitiv_Tensor_new_from_tensor(primitiv_Tensor *tensor) {
  return to_c(new Tensor(*to_cc(tensor)));
}

void primitiv_Tensor_delete(primitiv_Tensor *tensor) {
  delete to_cc(tensor);
}

bool primitiv_Tensor_valid(const primitiv_Tensor *tensor) {
  return to_cc(tensor)->valid();
}

primitiv_Status primitiv_Tensor_shape(const primitiv_Tensor *tensor,
                                      primitiv_Shape **shape) {
  try {
    *shape = to_c_from_value(to_cc(tensor)->shape());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_device(const primitiv_Tensor *tensor,
                                       primitiv_Device **device) {
  try {
    *device = to_c(&to_cc(tensor)->device());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_to_float(const primitiv_Tensor *tensor,
                                         float *value) {
  try {
    *value = to_cc(tensor)->to_float();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_to_array(const primitiv_Tensor *tensor,
                                         float *array) {
  try {
    std::vector<float> v = to_cc(tensor)->to_vector();
    std::copy(v.begin(), v.end(), array);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_argmax(const primitiv_Tensor *tensor,
                                       uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cc(tensor)->argmax(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_argmin(const primitiv_Tensor *tensor,
                                       uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cc(tensor)->argmin(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reset(primitiv_Tensor *tensor, float k) {
  try {
    to_cc(tensor)->reset(k);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                               const float *values) {
  try {
    to_cc(tensor)->reset_by_array(values);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reshape(const primitiv_Tensor *tensor,
                                        const primitiv_Shape *new_shape,
                                        primitiv_Tensor **new_tensor) {
  try {
    *new_tensor = to_c_from_value(to_cc(tensor)->reshape(*to_cc(new_shape)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_flatten(const primitiv_Tensor *tensor,
                                        primitiv_Tensor **new_tensor) {
  try {
    *new_tensor = to_c_from_value(to_cc(tensor)->flatten());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_multiply_const(primitiv_Tensor *tensor,
                                                       float k) {
  try {
    to_c(&(to_cc(tensor)->inplace_multiply_const(k)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_add(primitiv_Tensor *tensor,
                                            const primitiv_Tensor *x) {
  try {
    to_c(&(to_cc(tensor)->inplace_add(*to_cc(x))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_subtract(primitiv_Tensor *tensor,
                                                 const primitiv_Tensor *x) {
  try {
    to_c(&(to_cc(tensor)->inplace_subtract(*to_cc(x))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
