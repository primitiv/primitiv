#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/core/shape.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/shape.h>

using primitiv::Shape;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitivCreateShape(primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCreateShapeWithDims(
    const uint32_t *dims, size_t n, uint32_t batch,
    primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(dims);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Shape(std::vector<uint32_t>(dims, dims + n), batch));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCloneShape(
    const primitivShape_t *src, primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Shape(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteShape(primitivShape_t *shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  delete to_cpp_ptr(shape);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeDimSize(
    const primitivShape_t *shape, uint32_t i, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->operator[](i);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeDims(
    const primitivShape_t *shape, uint32_t *retval, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(shape)->dims(), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeDepth(
    const primitivShape_t *shape, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->depth();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeBatchSize(
    const primitivShape_t *shape, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->batch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeVolume(
    const primitivShape_t *shape, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->volume();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeLowerVolume(
    const primitivShape_t *shape, uint32_t dim, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->lower_volume(dim);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetShapeSize(
    const primitivShape_t *shape, uint32_t *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->size();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivRepresentShapeAsString(
    const primitivShape_t *shape, char *retval, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_string_to_array(
      to_cpp_ptr(shape)->to_string(), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsShapeEqualTo(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->operator==(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsNotShapeEqualTo(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->operator!=(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivHasShapeBatch(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->has_batch();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivHasShapeCompatibleBatch(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval =
      to_cpp_ptr(shape)->has_compatible_batch(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsShapeScalar(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->is_scalar();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsShapeColumnVector(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->is_column_vector();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsShapeMatrix(
    const primitivShape_t *shape, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->is_matrix();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivHasShapeSameDims(
    const primitivShape_t *shape, const primitivShape_t *rhs,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->has_same_dims(*to_cpp_ptr(rhs));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivHasShapeSameLooDims(
    const primitivShape_t *shape, const primitivShape_t *rhs, uint32_t dim,
    PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(rhs);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(shape)->has_same_loo_dims(*to_cpp_ptr(rhs), dim);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResizeShapeDim(
    const primitivShape_t *shape, uint32_t dim, uint32_t m,
    primitivShape_t **new_shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_dim(dim, m));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResizeShapeBatch(
    const primitivShape_t *shape, uint32_t batch,
    primitivShape_t **new_shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  *new_shape = to_c_ptr_from_value(to_cpp_ptr(shape)->resize_batch(batch));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivUpdateShapeDim(
    primitivShape_t *shape, uint32_t dim, uint32_t m) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(shape)->update_dim(dim, m);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivUpdateShapeBatchSize(
    primitivShape_t *shape, uint32_t batch) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  to_cpp_ptr(shape)->update_batch(batch);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
