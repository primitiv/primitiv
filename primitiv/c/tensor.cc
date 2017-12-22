/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <algorithm>
#include <vector>

#include <primitiv/tensor.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/tensor.h>

using primitiv::Tensor;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

extern "C" {

primitiv_Status primitiv_Tensor_new(primitiv_Tensor **tensor) try {
  *tensor = to_c_ptr(new Tensor());
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

primitiv_Status primitiv_Tensor_clone(
    primitiv_Tensor *src, primitiv_Tensor **tensor) try {
  *tensor = to_c_ptr(new Tensor(*to_cpp_ptr(src)));
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

void primitiv_Tensor_delete(primitiv_Tensor *tensor) {
  delete to_cpp_ptr(tensor);
}

bool primitiv_Tensor_valid(const primitiv_Tensor *tensor) {
  return to_cpp_ptr(tensor)->valid();
}

primitiv_Status primitiv_Tensor_shape(const primitiv_Tensor *tensor,
                                      primitiv_Shape **shape) {
  try {
    *shape = to_c_ptr_from_value(to_cpp_ptr(tensor)->shape());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_device(const primitiv_Tensor *tensor,
                                       primitiv_Device **device) {
  try {
    *device = to_c_ptr(&to_cpp_ptr(tensor)->device());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_to_float(const primitiv_Tensor *tensor,
                                         float *value) {
  try {
    *value = to_cpp_ptr(tensor)->to_float();
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_to_array(const primitiv_Tensor *tensor,
                                         float *array) {
  try {
    std::vector<float> v = to_cpp_ptr(tensor)->to_vector();
    std::copy(v.begin(), v.end(), array);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_argmax(const primitiv_Tensor *tensor,
                                       uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cpp_ptr(tensor)->argmax(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_argmin(const primitiv_Tensor *tensor,
                                       uint32_t dim, uint32_t *indices) {
  try {
    std::vector<uint32_t> v = to_cpp_ptr(tensor)->argmin(dim);
    std::copy(v.begin(), v.end(), indices);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reset(primitiv_Tensor *tensor, float k) {
  try {
    to_cpp_ptr(tensor)->reset(k);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reset_by_array(primitiv_Tensor *tensor,
                                               const float *values) {
  try {
    to_cpp_ptr(tensor)->reset_by_array(values);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_reshape(const primitiv_Tensor *tensor,
                                        const primitiv_Shape *new_shape,
                                        primitiv_Tensor **new_tensor) {
  try {
    *new_tensor = to_c_ptr_from_value(
        to_cpp_ptr(tensor)->reshape(*to_cpp_ptr(new_shape)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_flatten(const primitiv_Tensor *tensor,
                                        primitiv_Tensor **new_tensor) {
  try {
    *new_tensor = to_c_ptr_from_value(to_cpp_ptr(tensor)->flatten());
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_multiply_const(primitiv_Tensor *tensor,
                                                       float k) {
  try {
    to_c_ptr(&(to_cpp_ptr(tensor)->inplace_multiply_const(k)));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_add(primitiv_Tensor *tensor,
                                            const primitiv_Tensor *x) {
  try {
    to_c_ptr(&(to_cpp_ptr(tensor)->inplace_add(*to_cpp_ptr(x))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Tensor_inplace_subtract(primitiv_Tensor *tensor,
                                                 const primitiv_Tensor *x) {
  try {
    to_c_ptr(&(to_cpp_ptr(tensor)->inplace_subtract(*to_cpp_ptr(x))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
