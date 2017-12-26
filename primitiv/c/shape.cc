/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/shape.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/shape.h>

using primitiv::Shape;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

extern "C" {

primitiv_Status primitiv_Shape_new(primitiv_Shape **shape) try {
  *shape = to_c_ptr(new Shape());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_new_with_dims(
    const uint32_t *dims, size_t n, uint32_t batch,
    primitiv_Shape **shape) try {
  *shape = to_c_ptr(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

void primitiv_Shape_delete(primitiv_Shape *shape) {
  delete to_cpp_ptr(shape);
}

uint32_t primitiv_Shape_op_getitem(const primitiv_Shape *shape, uint32_t i) {
  return to_cpp_ptr(shape)->operator[](i);
}

void primitiv_Shape_dims(const primitiv_Shape *shape, uint32_t *array) {
  std::vector<uint32_t> v = to_cpp_ptr(shape)->dims();
  std::copy(v.begin(), v.end(), array);
}

uint32_t primitiv_Shape_depth(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->depth();
}

uint32_t primitiv_Shape_batch(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->batch();
}

uint32_t primitiv_Shape_volume(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->volume();
}

uint32_t primitiv_Shape_lower_volume(
    const primitiv_Shape *shape, uint32_t dim) {
  return to_cpp_ptr(shape)->lower_volume(dim);
}

uint32_t primitiv_Shape_size(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->size();
}

void primitiv_Shape_to_string(
    const primitiv_Shape *shape, char *string, size_t *length) {
  std::string str = to_cpp_ptr(shape)->to_string();
  *length = str.length();
  if (string) {
    std::strcpy(string, str.c_str());
  }
}

_Bool primitiv_Shape_op_eq(
    const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cpp_ptr(shape)->operator==(*to_cpp_ptr(rhs));
}

_Bool primitiv_Shape_op_ne(
    const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cpp_ptr(shape)->operator!=(*to_cpp_ptr(rhs));
}

_Bool primitiv_Shape_has_batch(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->has_batch();
}

_Bool primitiv_Shape_has_compatible_batch(
    const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cpp_ptr(shape)->has_compatible_batch(*to_cpp_ptr(rhs));
}

_Bool primitiv_Shape_is_scalar(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->is_scalar();
}

_Bool primitiv_Shape_is_row_vector(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->is_row_vector();
}

_Bool primitiv_Shape_is_matrix(const primitiv_Shape *shape) {
  return to_cpp_ptr(shape)->is_matrix();
}

_Bool primitiv_Shape_has_same_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs) {
  return to_cpp_ptr(shape)->has_same_dims(*to_cpp_ptr(rhs));
}

_Bool primitiv_Shape_has_same_loo_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim) {
  return to_cpp_ptr(shape)->has_same_loo_dims(*to_cpp_ptr(rhs), dim);
}

primitiv_Status primitiv_Shape_resize_dim(
    const primitiv_Shape *shape, uint32_t dim, uint32_t m,
    primitiv_Shape **new_shape) try {
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_dim(dim, m));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_resize_batch(
    const primitiv_Shape *shape, uint32_t batch,
    primitiv_Shape **new_shape) try {
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_batch(batch));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_update_dim(
    primitiv_Shape *shape, uint32_t dim, uint32_t m) try {
  to_cpp_ptr(shape)->update_dim(dim, m);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_update_batch(
    primitiv_Shape *shape, uint32_t batch) try {
  to_cpp_ptr(shape)->update_batch(batch);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
