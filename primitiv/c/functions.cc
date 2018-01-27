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

#define PRIMITIV_C_IMPL_UNARY_FUNC(name) \
PRIMITIV_C_STATUS primitiv_node_func_##name( \
    const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_tensor_func_##name( \
    const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(*to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \

#define PRIMITIV_C_IMPL_BINARY_OP(name) \
PRIMITIV_C_STATUS primitiv_node_func_##name##_node_const( \
    const primitivNode_t *x, float k, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_node_func_##name##_const_node( \
    float k, const primitivNode_t *x, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_node_func_##name##_node_node( \
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(a); \
  PRIMITIV_C_CHECK_NOT_NULL(b); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value( \
      primitiv::functions::name(*to_cpp_ptr(a), *to_cpp_ptr(b))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_tensor_func_##name##_tensor_const( \
    const primitivTensor_t *x, float k, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(*to_cpp_ptr(x), k)); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_tensor_func_##name##_const_tensor( \
    float k, const primitivTensor_t *x, primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(x); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value(primitiv::functions::name(k, *to_cpp_ptr(x))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \
PRIMITIV_C_STATUS primitiv_tensor_func_##name##_tensor_tensor( \
    const primitivTensor_t *a, const primitivTensor_t *b, \
    primitivTensor_t **y) try { \
  PRIMITIV_C_CHECK_NOT_NULL(a); \
  PRIMITIV_C_CHECK_NOT_NULL(b); \
  PRIMITIV_C_CHECK_NOT_NULL(y); \
  *y = to_c_ptr_from_value( \
      primitiv::functions::name(*to_cpp_ptr(a), *to_cpp_ptr(b))); \
  return PRIMITIV_C_OK; \
} PRIMITIV_C_HANDLE_EXCEPTIONS \

PRIMITIV_C_IMPL_UNARY_FUNC(positive);
PRIMITIV_C_IMPL_UNARY_FUNC(negative);
PRIMITIV_C_IMPL_BINARY_OP(add);
PRIMITIV_C_IMPL_BINARY_OP(subtract);
PRIMITIV_C_IMPL_BINARY_OP(multiply);
PRIMITIV_C_IMPL_BINARY_OP(divide);

PRIMITIV_C_STATUS primitiv_node_func_input(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(data);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::input_node(
      *to_cpp_ptr(shape), std::vector<float>(data, data + n), to_cpp_ptr(dev),
      to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_input(
    const primitivShape_t *shape, const float *data, size_t n,
    primitivDevice_t *dev, primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(data);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::input_tensor(
        *to_cpp_ptr(shape),
        std::vector<float>(data, data + n),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_parameter(
    primitivParameter_t *param, primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(
      primitiv::functions::parameter_node(*to_cpp_ptr(param), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_parameter(
    primitivParameter_t *param, primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(param);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::parameter_tensor(*to_cpp_ptr(param)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_copy(
    const primitivNode_t *x, primitivDevice_t *dev, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::copy(
        *to_cpp_ptr(x),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_copy(
    const primitivTensor_t *x, primitivDevice_t *dev, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::copy(
        *to_cpp_ptr(x),
        to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_pick(
    const primitivNode_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
     *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_pick(
    const primitivTensor_t *x, const uint32_t *ids, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(ids);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pick(
      *to_cpp_ptr(x), std::vector<uint32_t>(ids, ids + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_slice(
    const primitivNode_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_slice(
    const primitivTensor_t *x, uint32_t dim, uint32_t lower, uint32_t upper,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::slice(*to_cpp_ptr(x), dim, lower, upper));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_concat(
    const primitivNode_t *const *xs, size_t n, uint32_t dim,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Node*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_concat(
    const primitivTensor_t *const *xs, size_t n, uint32_t dim,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(primitiv::functions::concat(
      std::vector<const Tensor*>(_xs, _xs + n), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_reshape(
    const primitivNode_t *x, const primitivShape_t *new_shape,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::reshape(*to_cpp_ptr(x), *to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_reshape(
    const primitivTensor_t *x, const primitivShape_t *new_shape,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(new_shape);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::reshape(*to_cpp_ptr(x), *to_cpp_ptr(new_shape)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(flatten);
PRIMITIV_C_IMPL_UNARY_FUNC(transpose);

PRIMITIV_C_STATUS primitiv_node_func_matmul(
    const primitivNode_t *a, const primitivNode_t *b, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_matmul(
    const primitivTensor_t *a, const primitivTensor_t *b,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(a);
  PRIMITIV_C_CHECK_NOT_NULL(b);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::matmul(*to_cpp_ptr(a), *to_cpp_ptr(b)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(sqrt);
PRIMITIV_C_IMPL_UNARY_FUNC(exp);
PRIMITIV_C_IMPL_UNARY_FUNC(log);
PRIMITIV_C_IMPL_UNARY_FUNC(tanh);
PRIMITIV_C_IMPL_UNARY_FUNC(sigmoid);
PRIMITIV_C_IMPL_UNARY_FUNC(softplus);
PRIMITIV_C_IMPL_UNARY_FUNC(sin);
PRIMITIV_C_IMPL_UNARY_FUNC(cos);
PRIMITIV_C_IMPL_UNARY_FUNC(tan);
PRIMITIV_C_IMPL_UNARY_FUNC(relu);
PRIMITIV_C_IMPL_UNARY_FUNC(lrelu);

PRIMITIV_C_STATUS primitiv_node_func_prelu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_prelu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::prelu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_elu(
    const primitivNode_t *x, float a, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_elu(
    const primitivTensor_t *x, float a, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::elu(*to_cpp_ptr(x), a));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_sum(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_sum(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::sum(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_broadcast(
    const primitivNode_t *x, uint32_t dim, uint32_t size,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_broadcast(
    const primitivTensor_t *x, uint32_t dim, uint32_t size,
    primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::broadcast(*to_cpp_ptr(x), dim, size));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_logsumexp(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_logsumexp(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::logsumexp(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_log_softmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_log_softmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::log_softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_softmax(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_softmax(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::softmax(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_softmax_cross_entropy(
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

PRIMITIV_C_STATUS primitiv_tensor_func_softmax_cross_entropy(
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

PRIMITIV_C_STATUS primitiv_node_func_softmax_cross_entropy_with_array(
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

PRIMITIV_C_STATUS primitiv_tensor_func_softmax_cross_entropy_with_array(
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

PRIMITIV_C_IMPL_UNARY_FUNC(stop_gradient);

PRIMITIV_C_STATUS primitiv_node_func_batch_sum(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_batch_sum(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::sum(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_constant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::constant_node(
      *to_cpp_ptr(shape), k, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_constant(
    const primitivShape_t *shape, float k, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::constant_tensor(
          *to_cpp_ptr(shape), k, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_identity(
    uint32_t size, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(
      primitiv::functions::identity_node(
          size, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_identity(
    uint32_t size, primitivDevice_t *dev, primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::identity_tensor(size, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_random_bernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::random::bernoulli_node(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_random_bernoulli(
    const primitivShape_t *shape, float p, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(primitiv::functions::random::bernoulli_tensor(
      *to_cpp_ptr(shape), p, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_random_uniform(
    const primitivShape_t *shape, float lower, float upper, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::random::uniform_node(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_random_uniform(
    const primitivShape_t *shape, float lower, float upper, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(primitiv::functions::random::uniform_tensor(
      *to_cpp_ptr(shape), lower, upper, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_random_normal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::random::normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_random_normal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(primitiv::functions::random::normal_tensor(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_random_log_normal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::random::log_normal_node(
      *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_random_log_normal(
    const primitivShape_t *shape, float mean, float sd, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::random::log_normal_tensor(
          *to_cpp_ptr(shape), mean, sd, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_random_gumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivGraph_t *g, primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(primitiv::functions::random::gumbel_node(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_random_gumbel(
    const primitivShape_t *shape, float mu, float beta, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(primitiv::functions::random::gumbel_tensor(
      *to_cpp_ptr(shape), mu, beta, to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_BINARY_OP(pow);

PRIMITIV_C_STATUS primitiv_node_func_pown(
    const primitivNode_t *x, uint32_t k, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_pown(
    const primitivTensor_t *x, uint32_t k, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::pown(*to_cpp_ptr(x), k));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_IMPL_UNARY_FUNC(selu);

PRIMITIV_C_STATUS primitiv_node_func_sum_nodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_sum_tensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::sum(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_mean(
    const primitivNode_t *x, uint32_t dim, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_mean(
    const primitivTensor_t *x, uint32_t dim, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::mean(*to_cpp_ptr(x), dim));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_mean_nodes(
    const primitivNode_t *const *xs, size_t n, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Node*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_mean_tensors(
    const primitivTensor_t *const *xs, size_t n, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(xs);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  *y = to_c_ptr_from_value(
      primitiv::functions::mean(std::vector<const Tensor*>(_xs, _xs + n)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_batch_mean(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_batch_mean(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(primitiv::functions::batch::mean(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_batch_normalize(
    const primitivNode_t *x, primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_batch_normalize(
    const primitivTensor_t *x, primitivTensor_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::batch::normalize(*to_cpp_ptr(x)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_zeros(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(
      primitiv::functions::zeros_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_zeros(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::zeros_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_ones(
    const primitivShape_t *shape, primitivDevice_t *dev, primitivGraph_t *g,
    primitivNode_t **node) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(node);
  *node = to_c_ptr_from_value(
      primitiv::functions::ones_node(
          *to_cpp_ptr(shape), to_cpp_ptr(dev), to_cpp_ptr(g)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_ones(
    const primitivShape_t *shape, primitivDevice_t *dev,
    primitivTensor_t **tensor) try {
  PRIMITIV_C_CHECK_NOT_NULL(shape);
  PRIMITIV_C_CHECK_NOT_NULL(tensor);
  *tensor = to_c_ptr_from_value(
      primitiv::functions::ones_tensor(*to_cpp_ptr(shape), to_cpp_ptr(dev)));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_node_func_dropout(
    const primitivNode_t *x, float rate, PRIMITIV_C_BOOL enabled,
    primitivNode_t **y) try {
  PRIMITIV_C_CHECK_NOT_NULL(x);
  PRIMITIV_C_CHECK_NOT_NULL(y);
  *y = to_c_ptr_from_value(
      primitiv::functions::dropout(*to_cpp_ptr(x), rate, enabled));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitiv_tensor_func_dropout(
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
