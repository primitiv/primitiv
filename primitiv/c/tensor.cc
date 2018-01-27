#include <primitiv/config.h>

#include <vector>

#include <primitiv/tensor.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/tensor.h>

using primitiv::Tensor;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitiv_Tensor_new(primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(new Tensor());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_clone(
    primitivTensor_t *src, primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr(new Tensor(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_delete(primitivTensor_t *tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  delete to_cpp_ptr(tensor);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_valid(
    const primitivTensor_t *tensor, PRIMITIV_C_BOOL *valid) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(valid);
  *valid = to_cpp_ptr(tensor)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_shape(
    const primitivTensor_t *tensor, primitivShape_t **shape) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  *shape = to_c_ptr_from_value(to_cpp_ptr(tensor)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_device(
    const primitivTensor_t *tensor, primitivDevice_t **device) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(device);
  *device = to_c_ptr(&to_cpp_ptr(tensor)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_to_float(
    const primitivTensor_t *tensor, float *value) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(value);
  *value = to_cpp_ptr(tensor)->to_float();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_to_array(
    const primitivTensor_t *tensor, float *array, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->to_vector(), array, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_argmax(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->argmax(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_argmin(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *indices,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->argmin(dim), indices, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_reset(primitivTensor_t *tensor, float k) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  to_cpp_ptr(tensor)->reset(k);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_reset_by_array(
    primitivTensor_t *tensor, const float *values) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(values);
  to_cpp_ptr(tensor)->reset_by_array(values);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_reshape(
    const primitivTensor_t *tensor, const primitivShape_t *new_shape,
    primitivTensor_t **new_tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(new_tensor);
  *new_tensor = to_c_ptr_from_value(
      to_cpp_ptr(tensor)->reshape(*to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_flatten(
    const primitivTensor_t *tensor, primitivTensor_t **new_tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(new_tensor);
  *new_tensor = to_c_ptr_from_value(to_cpp_ptr(tensor)->flatten());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_inplace_multiply_const(
    primitivTensor_t *tensor, float k) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_multiply_const(k)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_inplace_add(
    primitivTensor_t *tensor, const primitivTensor_t *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_add(*to_cpp_ptr(x))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_Tensor_inplace_subtract(
    primitivTensor_t *tensor, const primitivTensor_t *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_subtract(*to_cpp_ptr(x))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
