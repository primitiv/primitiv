#include <primitiv/config.h>

#include <vector>

#include <primitiv/functions.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/functions.h>

using primitiv::Node;
using primitiv::Tensor;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;
using primitiv::c::internal::to_c_ptr_from_value;

#define PRIMITIV_C_IMPL_UNARY_FUNC(name, cpp_func) \
PRIMITIV_C_STATUS primitivNode##name( \
    const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivTensor##name( \
    const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \

#define PRIMITIV_C_IMPL_BINARY_OP(name, cpp_func) \
PRIMITIV_C_STATUS primitivNode##name##NodeConst( \
    const primitivNode_t *x, float k, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivNode##name##ConstNode( \
    float k, const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivNode##name##NodeNode( \
    const primitivNode_t *a, const primitivNode_t *b, \
    primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(a); \
  PRIMITIV_C_CHECK_NOT_NULL(b); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value( \
      primitiv::functions::cpp_func(*to_cpp_ptr(a), *to_cpp_ptr(b))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivTensor##name##TensorConst( \
    const primitivTensor_t *x, float k, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivTensor##name##ConstTensor( \
    float k, const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::cpp_func(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitivTensor##name##TensorTensor( \
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

PRIMITIV_C_STATUS primitivNodeInput(
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

PRIMITIV_C_STATUS primitivTensorInput(
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

PRIMITIV_C_STATUS primitivNodeParameter(
    primitivParameter_t *param, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::parameter_node(*to_cpp_ptr(param), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorParameter(
    primitivParameter_t *param, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::parameter_tensor(*to_cpp_ptr(param)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeCopy(
    const primitivNode_t *x, primitivDevice_t *dev, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::copy(
        *to_cpp_ptr(x),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorCopy(
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

PRIMITIV_C_STATUS primitivNodePick(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
     *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorPick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
      *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeSlice(
    const primitivNode_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorSlice(
    const primitivTensor_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeConcat(
    const primitivNode_t *const *xs, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Node*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorConcat(
    const primitivTensor_t *const *xs, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Tensor*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeReshape(
    const primitivNode_t *x, const primitivShape_t *new_shape,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::reshape(*to_cpp_ptr(x), *to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorReshape(
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

PRIMITIV_C_STATUS primitivNodeMatmul(
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorMatmul(
    const primitivTensor_t *a, const primitivTensor_t *b,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

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

PRIMITIV_C_STATUS primitivNodePrelu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorPrelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeElu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensoRelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeSum(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorSum(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeBroadcast(
    const primitivNode_t *x, uint32_t dim, uint32_t size,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorBroadcast(
    const primitivTensor_t *x, uint32_t dim, uint32_t size,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeLogsumexp(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorLogsumexp(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeLogSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorLogSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeSoftmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorSoftmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeSoftmaxCrossEntropy(
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

PRIMITIV_C_STATUS primitivTensorSoftmaxCrossEntropy(
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

PRIMITIV_C_STATUS primitivNodeSoftmaxCrossEntropyWithArray(
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

PRIMITIV_C_STATUS primitivTensorSoftmaxCrossEntropyWithArray(
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

PRIMITIV_C_STATUS primitivNodeBatchSum(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorBatchSum(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::constant_node(
      *to_cpp_ptr(shape), k, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorConstant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::constant_tensor(
          *to_cpp_ptr(shape), k, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeIdentity(
    uint32_t size, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::identity_node(
          size, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorIdentity(
    uint32_t size, primitivDevice_t *dev, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::identity_tensor(size, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::bernoulli_node(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorRandomBernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::bernoulli_tensor(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::uniform_node(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorRandomUniform(
    const primitivShape_t *shape, float lower, float upper,
    primitivDevice_t *dev, primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::uniform_tensor(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorRandomNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::normal_tensor(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::log_normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorRandomLogNormal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::random::log_normal_tensor(
          *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::gumbel_node(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorRandomGumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(primitiv::functions::random::gumbel_tensor(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_BINARY_OP(Pow, pow);

PRIMITIV_C_STATUS primitivNodePown(
    const primitivNode_t *x, uint32_t k, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorPown(
    const primitivTensor_t *x, uint32_t k, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(Selu, selu);

PRIMITIV_C_STATUS primitivNodeSumNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorSumTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeMean(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorMean(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeMeanNodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorMeanTensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeBatchMean(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorBatchMean(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeBatchNormalize(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorBatchNormalize(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeZeros(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::zeros_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorZeros(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::zeros_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeOnes(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::ones_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorOnes(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      primitiv::functions::ones_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivNodeDropout(
    const primitivNode_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::dropout(*to_cpp_ptr(x), rate, enabled));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivTensorDropout(
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
