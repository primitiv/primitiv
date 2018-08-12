#include <primitiv/config.h>

#include <vector>

#include <primitiv/core/functions.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/functions.h>

using primitiv::Node;
using primitiv::Tensor;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

#define PRIMITIV_C_IMPL_UNARY_FUNC(name, cpp_func) \
PRIMITIV_C_STATUS primitivApplyNode##name( \
    const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyTensor##name( \
    const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \

#define PRIMITIV_C_IMPL_BINARY_OP(name, cpp_func) \
PRIMITIV_C_STATUS primitivApplyNode##name##XC( \
    const primitivNode_t *x, float k, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyNode##name##CX( \
    float k, const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyNode##name( \
    const primitivNode_t *a, const primitivNode_t *b, \
    primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(a); \
  PRIMITIV_C_CHECK_NOT_NULL(b); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value( \
      primitiv::functions::cpp_func(*to_cpp_ptr(a), *to_cpp_ptr(b))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyTensor##name##XC( \
    const primitivTensor_t *x, float k, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyTensor##name##CX( \
    float k, const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivApplyTensor##name( \
    const primitivTensor_t *a, const primitivTensor_t *b, \
    primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(a); \
  PRIMITIV_C_CHECK_NOT_NULL(b); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value( \
      primitiv::functions::cpp_func(*to_cpp_ptr(a), *to_cpp_ptr(b))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \

PRIMITIV_C_IMPL_UNARY_FUNC(Positive, positive);
PRIMITIV_C_IMPL_UNARY_FUNC(Negative, negative);
PRIMITIV_C_IMPL_BINARY_OP(Add, add);
PRIMITIV_C_IMPL_BINARY_OP(Subtract, subtract);
PRIMITIV_C_IMPL_BINARY_OP(Multiply, multiply);
PRIMITIV_C_IMPL_BINARY_OP(Divide, divide);
PRIMITIV_C_IMPL_BINARY_OP(Pow, pow);

PRIMITIV_C_STATUS primitivApplyNodePowN(
    const primitivNode_t *x, int32_t k, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorPowN(
    const primitivTensor_t *x, int32_t k, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(data);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::input_node(
      *to_cpp_ptr(shape), std::vector<float>(data, data + n), to_cpp_ptr(dev),
      to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorInput(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(data);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::input_tensor(
        *to_cpp_ptr(shape),
        std::vector<float>(data, data + n),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeParameter(
    primitivParameter_t *param, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::parameter_node(*to_cpp_ptr(param), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorParameter(
    primitivParameter_t *param, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::parameter_tensor(*to_cpp_ptr(param)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeCopy(
    const primitivNode_t *x, primitivDevice_t *dev, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::copy(
        *to_cpp_ptr(x),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorCopy(
    const primitivTensor_t *x, primitivDevice_t *dev,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::copy(
        *to_cpp_ptr(x),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodePick(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
     *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
      *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSlice(
    const primitivNode_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSlice(
    const primitivTensor_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSplit(
    const primitivNode_t *x, uint32_t dim, uint32_t n, primitivNode_t **ys) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ys);
  std::vector<primitiv::Node> nodes =
      primitiv::functions::split(*to_cpp_ptr(x), dim, n);
  size_t size = n;
  primitiv::c::internal::move_vector_to_array_of_c_ptrs(&nodes, ys, &size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSplit(
    const primitivTensor_t *x, uint32_t dim, uint32_t n, primitivTensor_t **ys) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ys);
  std::vector<primitiv::Tensor> tensors =
      primitiv::functions::split(*to_cpp_ptr(x), dim, n);
  size_t size = n;
  primitiv::c::internal::move_vector_to_array_of_c_ptrs(&tensors, ys, &size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeConcat(
    const primitivNode_t *const *xs, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Node*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorConcat(
    const primitivTensor_t *const *xs, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Tensor*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeReshape(
    const primitivNode_t *x, const primitivShape_t *new_shape,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::reshape(*to_cpp_ptr(x), *to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorReshape(
    const primitivTensor_t *x, const primitivShape_t *new_shape,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::reshape(*to_cpp_ptr(x), *to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(Flatten, flatten);
PRIMITIV_C_IMPL_UNARY_FUNC(Transpose, transpose);

PRIMITIV_C_STATUS primitivApplyNodeFlip(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::flip(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorFlip(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::flip(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodePermuteDims(
    const primitivNode_t *x, const uint32_t *perm, size_t n,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(perm);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::permute_dims(
     *to_cpp_ptr(x), std::vector<uint32_t>(perm, perm + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorPermuteDims(
    const primitivTensor_t *x, const uint32_t *perm, size_t n,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(perm);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::permute_dims(
      *to_cpp_ptr(x), std::vector<uint32_t>(perm, perm + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMatmul(
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMatmul(
    const primitivTensor_t *a, const primitivTensor_t *b,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(Abs, abs);
PRIMITIV_C_IMPL_UNARY_FUNC(Sqrt, sqrt);
PRIMITIV_C_IMPL_UNARY_FUNC(Exp, exp);
PRIMITIV_C_IMPL_UNARY_FUNC(Log, log);
PRIMITIV_C_IMPL_UNARY_FUNC(Tanh, tanh);
PRIMITIV_C_IMPL_UNARY_FUNC(Sigmoid, sigmoid);
PRIMITIV_C_IMPL_UNARY_FUNC(Softplus, softplus);
PRIMITIV_C_IMPL_UNARY_FUNC(Sin, sin);
PRIMITIV_C_IMPL_UNARY_FUNC(Cos, cos);
PRIMITIV_C_IMPL_UNARY_FUNC(Tan, tan);
PRIMITIV_C_IMPL_UNARY_FUNC(Relu, relu);
PRIMITIV_C_IMPL_UNARY_FUNC(Lrelu, lrelu);

PRIMITIV_C_STATUS primitivApplyNodePrelu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorPrelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeElu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorElu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::max(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::max(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMin(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::min(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMin(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::min(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSum(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSum(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBroadcast(
    const primitivNode_t *x, uint32_t dim, uint32_t size,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBroadcast(
    const primitivTensor_t *x, uint32_t dim, uint32_t size,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeLogsumexp(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorLogsumexp(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeLogSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorLogSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSoftmaxCrossEntropy(
    const primitivNode_t *x, const primitivNode_t *t, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(t);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cpp_ptr(x), *to_cpp_ptr(t), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSoftmaxCrossEntropy(
    const primitivTensor_t *x, const primitivTensor_t *t, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(t);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cpp_ptr(x), *to_cpp_ptr(t), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeSoftmaxCrossEntropyWithArray(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSoftmaxCrossEntropyWithArray(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(StopGradient, stop_gradient);

PRIMITIV_C_STATUS primitivApplyNodeConv2d(
    const primitivNode_t *x, const primitivNode_t *w,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    uint32_t dilation0, uint32_t dilation1,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(w);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::conv2d(
      *to_cpp_ptr(x), *to_cpp_ptr(w),
      padding0, padding1, stride0, stride1, dilation0, dilation1));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorConv2d(
    const primitivTensor_t *x, const primitivTensor_t *w,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    uint32_t dilation0, uint32_t dilation1,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(w);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::conv2d(
      *to_cpp_ptr(x), *to_cpp_ptr(w),
      padding0, padding1, stride0, stride1, dilation0, dilation1));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMaxPool2d(
    const primitivNode_t *x,
    uint32_t window0, uint32_t window1,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::max_pool2d(
      *to_cpp_ptr(x), window0, window1, padding0, padding1, stride0, stride1));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMaxPool2d(
    const primitivTensor_t *x,
    uint32_t window0, uint32_t window1,
    uint32_t padding0, uint32_t padding1,
    uint32_t stride0, uint32_t stride1,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::max_pool2d(
      *to_cpp_ptr(x), window0, window1, padding0, padding1, stride0, stride1));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchPick(
    const primitivNode_t *x, const uint32_t *ids, size_t n,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::pick(
     *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::pick(
     *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchSlice(
    const primitivNode_t *x, uint32_t lower, uint32_t upper,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::slice(*to_cpp_ptr(x), lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchSlice(
    const primitivTensor_t *x, uint32_t lower, uint32_t upper,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::slice(*to_cpp_ptr(x), lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchSplit(
    const primitivNode_t *x, uint32_t n, primitivNode_t **ys) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ys);
  std::vector<primitiv::Node> nodes =
      primitiv::functions::batch::split(*to_cpp_ptr(x), n);
  size_t size = n;
  primitiv::c::internal::move_vector_to_array_of_c_ptrs(&nodes, ys, &size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchSplit(
    const primitivTensor_t *x, uint32_t n, primitivTensor_t **ys) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ys);
  std::vector<primitiv::Tensor> nodes =
      primitiv::functions::batch::split(*to_cpp_ptr(x), n);
  size_t size = n;
  primitiv::c::internal::move_vector_to_array_of_c_ptrs(&nodes, ys, &size);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchConcat(
    const primitivNode_t *const *xs, size_t n,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::batch::concat(
      std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchConcat(
    const primitivTensor_t *const *xs, size_t n,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::batch::concat(
      std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchSum(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchSum(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::constant_node(
      *to_cpp_ptr(shape), k, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::constant_tensor(
          *to_cpp_ptr(shape), k, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeIdentity(
    uint32_t size, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::identity_node(
          size, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorIdentity(
    uint32_t size, primitivDevice_t *dev, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::identity_tensor(size, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::bernoulli_node(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::bernoulli_tensor(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::uniform_node(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::uniform_tensor(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::normal_tensor(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::log_normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::random::log_normal_tensor(
          *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::gumbel_node(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::gumbel_tensor(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(Selu, selu);

PRIMITIV_C_STATUS primitivApplyNodeSumNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorSumTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMean(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMean(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeMeanNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorMeanTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchMean(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchMean(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeBatchNormalize(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorBatchNormalize(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeZeros(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::zeros_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorZeros(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::zeros_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeOnes(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::ones_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorOnes(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::ones_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyNodeDropout(
    const primitivNode_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::dropout(*to_cpp_ptr(x), rate, enabled));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivApplyTensorDropout(
    const primitivTensor_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::dropout(*to_cpp_ptr(x), rate, enabled));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

#undef PRIMITIV_C_IMPL_UNARY_FUNC
#undef PRIMITIV_C_IMPL_BINARY_OP
