#include <primitiv/config.h>

#include <vector>

#include <primitiv/core/tensor.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/tensor.h>

using primitiv::Tensor;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

PRIMITIV_C_STATUS primitivCreateTensor(primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Tensor());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivCloneTensor(
    const primitivTensor_t *src, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(src);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Tensor(*to_cpp_ptr(src)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteTensor(primitivTensor_t *tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  delete to_cpp_ptr(tensor);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivIsValidTensor(
    const primitivTensor_t *tensor, PRIMITIV_C_BOOL *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(tensor)->valid();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetTensorShape(
    const primitivTensor_t *tensor, primitivShape_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(tensor)->shape());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetDeviceFromTensor(
    const primitivTensor_t *tensor, primitivDevice_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&to_cpp_ptr(tensor)->device());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateTensorAsFloat(
    const primitivTensor_t *tensor, float *retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(tensor)->to_float();
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivEvaluateTensorAsArray(
    const primitivTensor_t *tensor, float *retval, size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->to_vector(), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetTensorArgmax(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *retval,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->argmax(dim), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetTensorArgmin(
    const primitivTensor_t *tensor, uint32_t dim, uint32_t *retval,
    size_t *size) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(size);
  primitiv::c::internal::copy_vector_to_array(
      to_cpp_ptr(tensor)->argmin(dim), retval, size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResetTensor(primitivTensor_t *tensor, float k) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  to_cpp_ptr(tensor)->reset(k);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivResetTensorByArray(
    primitivTensor_t *tensor, const float *values) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(values);
  to_cpp_ptr(tensor)->reset_by_array(values);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivReshapeTensor(
    const primitivTensor_t *tensor, const primitivShape_t *new_shape,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(tensor)->reshape(*to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivFlattenTensor(
    const primitivTensor_t *tensor, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(tensor)->flatten());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivMultiplyTensorByConstantInplace(
    primitivTensor_t *tensor, float k) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_multiply_const(k)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddTensorInplace(
    primitivTensor_t *tensor, const primitivTensor_t *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_add(*to_cpp_ptr(x))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSubtractTensorInplace(
    primitivTensor_t *tensor, const primitivTensor_t *x) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  PRIMITIV_C_CHECK_NOT_NULL(x);
  to_c_ptr(&(to_cpp_ptr(tensor)->inplace_subtract(*to_cpp_ptr(x))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
