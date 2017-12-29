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
  PRIMITIV_C_CHECK_PTR_ARG(dims);
  *shape = to_c_ptr(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_delete(primitiv_Shape *shape) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  delete to_cpp_ptr(shape);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_op_getitem(
    const primitiv_Shape *shape, uint32_t i, uint32_t *dim_size) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *dim_size = to_cpp_ptr(shape)->operator[](i);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_dims(
    const primitiv_Shape *shape, uint32_t *dims, size_t *array_size) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  const std::vector<uint32_t> v = to_cpp_ptr(shape)->dims();
  if (array_size) {
    *array_size = v.size();
  }
  if (dims) {
    std::copy(v.begin(), v.end(), dims);
  }
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_depth(
    const primitiv_Shape *shape, uint32_t *depth) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *depth = to_cpp_ptr(shape)->depth();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_batch(
    const primitiv_Shape *shape, uint32_t *batch) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *batch = to_cpp_ptr(shape)->batch();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_volume(
    const primitiv_Shape *shape, uint32_t *volume) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *volume = to_cpp_ptr(shape)->volume();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_lower_volume(
    const primitiv_Shape *shape, uint32_t dim, uint32_t *lower_volume) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *lower_volume = to_cpp_ptr(shape)->lower_volume(dim);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_size(
    const primitiv_Shape *shape, uint32_t *size) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *size = to_cpp_ptr(shape)->size();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_to_string(
    const primitiv_Shape *shape, char *buffer, size_t *buffer_size) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  const std::string str = to_cpp_ptr(shape)->to_string();
  if (buffer_size) {
    *buffer_size = str.length();
  }
  if (buffer) {
    std::strcpy(buffer, str.c_str());
  }
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_op_eq(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    unsigned char *eq) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(rhs);
  *eq = to_cpp_ptr(shape)->operator==(*to_cpp_ptr(rhs));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_op_ne(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    unsigned char *ne) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(rhs);
  *ne = to_cpp_ptr(shape)->operator!=(*to_cpp_ptr(rhs));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_has_batch(
    const primitiv_Shape *shape, unsigned char *has_batch) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *has_batch = to_cpp_ptr(shape)->has_batch();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_has_compatible_batch(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    unsigned char *has_compatible_batch) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(rhs);
  *has_compatible_batch =
      to_cpp_ptr(shape)->has_compatible_batch(*to_cpp_ptr(rhs));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_is_scalar(
    const primitiv_Shape *shape, unsigned char *is_scalar) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *is_scalar = to_cpp_ptr(shape)->is_scalar();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_is_row_vector(
    const primitiv_Shape *shape, unsigned char *is_row_vector) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *is_row_vector = to_cpp_ptr(shape)->is_row_vector();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_is_matrix(
    const primitiv_Shape *shape, unsigned char *is_matrix) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *is_matrix = to_cpp_ptr(shape)->is_matrix();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_has_same_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    unsigned char *has_same_dims) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(rhs);
  *has_same_dims = to_cpp_ptr(shape)->has_same_dims(*to_cpp_ptr(rhs));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_has_same_loo_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim,
    unsigned char *has_same_loo_dims) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  PRIMITIV_C_CHECK_PTR_ARG(rhs);
  *has_same_loo_dims =
      to_cpp_ptr(shape)->has_same_loo_dims(*to_cpp_ptr(rhs), dim);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_resize_dim(
    const primitiv_Shape *shape, uint32_t dim, uint32_t m,
    primitiv_Shape **new_shape) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_dim(dim, m));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_resize_batch(
    const primitiv_Shape *shape, uint32_t batch,
    primitiv_Shape **new_shape) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_batch(batch));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_update_dim(
    primitiv_Shape *shape, uint32_t dim, uint32_t m) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  to_cpp_ptr(shape)->update_dim(dim, m);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Shape_update_batch(
    primitiv_Shape *shape, uint32_t batch) try {
  PRIMITIV_C_CHECK_PTR_ARG(shape);
  to_cpp_ptr(shape)->update_batch(batch);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
