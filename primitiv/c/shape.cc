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

PRIMITIV_C_STATUS primitiv_Shape_new(primitiv_Shape **shape) try {
  *shape = to_c_ptr(new Shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_new_with_dims(
    const uint32_t *dims, size_t n, uint32_t batch,
    primitiv_Shape **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(dims);
  *shape = to_c_ptr(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_delete(primitiv_Shape *shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  delete to_cpp_ptr(shape);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_op_getitem(
    const primitiv_Shape *shape, uint32_t i, uint32_t *dim_size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *dim_size = to_cpp_ptr(shape)->operator[](i);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_dims(
    const primitiv_Shape *shape, uint32_t *dims, size_t *array_size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(array_size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(shape)->dims(), dims, array_size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_depth(
    const primitiv_Shape *shape, uint32_t *depth) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *depth = to_cpp_ptr(shape)->depth();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_batch(
    const primitiv_Shape *shape, uint32_t *batch) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *batch = to_cpp_ptr(shape)->batch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_volume(
    const primitiv_Shape *shape, uint32_t *volume) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *volume = to_cpp_ptr(shape)->volume();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_lower_volume(
    const primitiv_Shape *shape, uint32_t dim, uint32_t *lower_volume) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *lower_volume = to_cpp_ptr(shape)->lower_volume(dim);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_size(
    const primitiv_Shape *shape, uint32_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *size = to_cpp_ptr(shape)->size();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_to_string(
    const primitiv_Shape *shape, char *buffer, size_t *buffer_size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(buffer_size);
  primitiv::c::internal::copy_string_to_array(
      to_cpp_ptr(shape)->to_string(), buffer, buffer_size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_op_eq(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    PRIMITIV_C_BOOL *eq) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  *eq = to_cpp_ptr(shape)->operator==(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_op_ne(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    PRIMITIV_C_BOOL *ne) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  *ne = to_cpp_ptr(shape)->operator!=(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_has_batch(
    const primitiv_Shape *shape, PRIMITIV_C_BOOL *has_batch) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *has_batch = to_cpp_ptr(shape)->has_batch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_has_compatible_batch(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    PRIMITIV_C_BOOL *has_compatible_batch) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  *has_compatible_batch =
      to_cpp_ptr(shape)->has_compatible_batch(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_is_scalar(
    const primitiv_Shape *shape, PRIMITIV_C_BOOL *is_scalar) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *is_scalar = to_cpp_ptr(shape)->is_scalar();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_is_column_vector(
    const primitiv_Shape *shape, PRIMITIV_C_BOOL *is_column_vector) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *is_column_vector = to_cpp_ptr(shape)->is_column_vector();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_is_matrix(
    const primitiv_Shape *shape, PRIMITIV_C_BOOL *is_matrix) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *is_matrix = to_cpp_ptr(shape)->is_matrix();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_has_same_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs,
    PRIMITIV_C_BOOL *has_same_dims) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  *has_same_dims = to_cpp_ptr(shape)->has_same_dims(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_has_same_loo_dims(
    const primitiv_Shape *shape, const primitiv_Shape *rhs, uint32_t dim,
    PRIMITIV_C_BOOL *has_same_loo_dims) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  *has_same_loo_dims =
      to_cpp_ptr(shape)->has_same_loo_dims(*to_cpp_ptr(rhs), dim);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_resize_dim(
    const primitiv_Shape *shape, uint32_t dim, uint32_t m,
    primitiv_Shape **new_shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_dim(dim, m));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_resize_batch(
    const primitiv_Shape *shape, uint32_t batch,
    primitiv_Shape **new_shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_batch(batch));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_update_dim(
    primitiv_Shape *shape, uint32_t dim, uint32_t m) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(shape)->update_dim(dim, m);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Shape_update_batch(
    primitiv_Shape *shape, uint32_t batch) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(shape)->update_batch(batch);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
