/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <vector>

#include <primitiv/functions.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/functions.h>

using primitiv::Node;
using primitiv::Tensor;

extern "C" {

#define IMPL_UNARY_FUNC(name) \
primitiv_Node *_NODE_FUN(name)(const primitiv_Node *x) { \
  return to_c_from_value(primitiv::functions::name(*to_cc(x))); \
} \
primitiv_Node *_S_NODE_FUN(name)(const primitiv_Node *x, \
                                 primitiv_Status *status) { \
  SAFE_RETURN(_NODE_FUN(name)(x), status, nullptr); \
} \
primitiv_Tensor *_TENSOR_FUN(name)(const primitiv_Tensor *x) { \
  return to_c_from_value(primitiv::functions::name(*to_cc(x))); \
} \
primitiv_Tensor *_S_TENSOR_FUN(name)(const primitiv_Tensor *x, \
                                     primitiv_Status *status) { \
  SAFE_RETURN(_TENSOR_FUN(name)(x), status, nullptr); \
}

#define IMPL_BINARY_OPERATOR(op_name) \
primitiv_Node *_NODE_OP(op_name, _node_const)( \
    const primitiv_Node *x, float k) { \
  return to_c_from_value(primitiv::functions::op_name(*to_cc(x), k)); \
} \
primitiv_Node *_S_NODE_OP(op_name, _node_const)( \
    const primitiv_Node *x, float k, primitiv_Status *status) { \
  SAFE_RETURN(_NODE_OP(op_name, _node_const)(x, k), status, nullptr); \
} \
primitiv_Node *_NODE_OP(op_name, _const_node)( \
    float k, const primitiv_Node *x) { \
  return to_c_from_value(primitiv::functions::op_name(k, *to_cc(x))); \
} \
primitiv_Node *_S_NODE_OP(op_name, _const_node)( \
    float k, const primitiv_Node *x, primitiv_Status *status) { \
  SAFE_RETURN(_NODE_OP(op_name, _const_node)(k, x), status, nullptr); \
} \
primitiv_Node *_NODE_OP(op_name, _node_node)( \
    const primitiv_Node *a, const primitiv_Node *b) { \
  return to_c_from_value(primitiv::functions::op_name(*to_cc(a), *to_cc(b))); \
} \
primitiv_Node *_S_NODE_OP(op_name, _node_node)( \
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status) { \
  SAFE_RETURN(_NODE_OP(op_name, _node_node)(a, b), status, nullptr); \
} \
primitiv_Tensor *_TENSOR_OP(op_name, _tensor_const)( \
    const primitiv_Tensor *x, float k) { \
  return to_c_from_value(primitiv::functions::op_name(*to_cc(x), k)); \
} \
primitiv_Tensor *_S_TENSOR_OP(op_name, _tensor_const)( \
    const primitiv_Tensor *x, float k, primitiv_Status *status) { \
  SAFE_RETURN(_TENSOR_OP(op_name, _tensor_const)(x, k), status, nullptr); \
} \
primitiv_Tensor *_TENSOR_OP(op_name, _const_tensor)( \
    float k, const primitiv_Tensor *x) { \
  return to_c_from_value(primitiv::functions::op_name(k, *to_cc(x))); \
} \
primitiv_Tensor *_S_TENSOR_OP(op_name, _const_tensor)( \
    float k, const primitiv_Tensor *x, primitiv_Status *status) { \
  SAFE_RETURN(_TENSOR_OP(op_name, _const_tensor)(k, x), status, nullptr); \
} \
primitiv_Tensor *_TENSOR_OP(op_name, _tensor_tensor)( \
    const primitiv_Tensor *a, const primitiv_Tensor *b) { \
  return to_c_from_value(primitiv::functions::op_name(*to_cc(a), *to_cc(b))); \
} \
primitiv_Tensor *_S_TENSOR_OP(op_name, _tensor_tensor)( \
    const primitiv_Tensor *a, \
    const primitiv_Tensor *b, \
    primitiv_Status *status) { \
  SAFE_RETURN(_TENSOR_OP(op_name, _tensor_tensor)(a, b), status, nullptr); \
}

IMPL_UNARY_FUNC(positive);
IMPL_UNARY_FUNC(negative);
IMPL_BINARY_OPERATOR(add);
IMPL_BINARY_OPERATOR(subtract);
IMPL_BINARY_OPERATOR(multiply);
IMPL_BINARY_OPERATOR(divide);

primitiv_Node *primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::input_node(
      *to_cc(shape), std::vector<float>(data, data + n), to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_node_func_input(shape, data, n, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::input_tensor(
      *to_cc(shape), std::vector<float>(data, data + n), to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_input(shape, data, n, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_parameter(
    primitiv_Parameter *param, primitiv_Graph *g) {
  return to_c_from_value(
      primitiv::functions::parameter_node(*to_cc(param), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_parameter(
    primitiv_Parameter *param, primitiv_Graph *g, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_parameter(param, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_parameter(primitiv_Parameter *param) {
  return to_c_from_value(primitiv::functions::parameter_tensor(*to_cc(param)));
}
primitiv_Tensor *safe_primitiv_tensor_func_parameter(
    primitiv_Parameter *param, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_parameter(param), status, nullptr);
}

primitiv_Node *primitiv_node_func_copy(const primitiv_Node *x,
                                       primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::copy(*to_cc(x), to_cc(dev)));
}
primitiv_Node *safe_primitiv_node_func_copy(const primitiv_Node *x,
                                            primitiv_Device *dev,
                                            primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_copy(x, dev), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_copy(const primitiv_Tensor *x,
                                           primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::copy(*to_cc(x), to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_copy(const primitiv_Tensor *x,
                                                primitiv_Device *dev,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_copy(x, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_pick(const primitiv_Node *x,
                                       const uint32_t *ids,
                                       size_t n,
                                       uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::pick(*to_cc(x),
                                std::vector<uint32_t>(ids, ids + n),
                                dim));
}
primitiv_Node *safe_primitiv_node_func_pick(const primitiv_Node *x,
                                            const uint32_t *ids,
                                            size_t n,
                                            uint32_t dim,
                                            primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_pick(x, ids, n, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_pick(const primitiv_Tensor *x,
                                           const uint32_t *ids,
                                           size_t n,
                                           uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::pick(*to_cc(x),
                                std::vector<uint32_t>(ids, ids + n),
                                dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_pick(const primitiv_Tensor *x,
                                                const uint32_t *ids,
                                                size_t n,
                                                uint32_t dim,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_pick(x, ids, n, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_slice(const primitiv_Node *x,
                                        uint32_t dim,
                                        uint32_t lower,
                                        uint32_t upper) {
  return to_c_from_value(
      primitiv::functions::slice(*to_cc(x), dim, lower, upper));
}
primitiv_Node *safe_primitiv_node_func_slice(const primitiv_Node *x,
                                             uint32_t dim,
                                             uint32_t lower,
                                             uint32_t upper,
                                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_slice(x, dim, lower, upper), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_slice(const primitiv_Tensor *x,
                                            uint32_t dim,
                                            uint32_t lower,
                                            uint32_t upper) {
  return to_c_from_value(
      primitiv::functions::slice(*to_cc(x), dim, lower, upper));
}
primitiv_Tensor *safe_primitiv_tensor_func_slice(const primitiv_Tensor *x,
                                                 uint32_t dim,
                                                 uint32_t lower,
                                                 uint32_t upper,
                                                 primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_slice(x, dim, lower, upper), status, nullptr);
}

primitiv_Node *primitiv_node_func_concat(const primitiv_Node *const *xs,
                                         size_t n,
                                         uint32_t dim) {
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  return to_c_from_value(primitiv::functions::concat(
      std::vector<const Node*>(_xs, _xs + n), dim));
}
primitiv_Node *safe_primitiv_node_func_concat(const primitiv_Node *const *xs,
                                              size_t n,
                                              uint32_t dim,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_concat(xs, n, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_concat(
    const primitiv_Tensor *const *xs,
    size_t n,
    uint32_t dim) {
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  return to_c_from_value(primitiv::functions::concat(
      std::vector<const Tensor*>(_xs, _xs + n), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_concat(
    const primitiv_Tensor *const *xs,
    size_t n,
    uint32_t dim,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_concat(xs, n, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_reshape(
    const primitiv_Node *x,
    const primitiv_Shape *new_shape) {
  return to_c_from_value(
      primitiv::functions::reshape(*to_cc(x), *to_cc(new_shape)));
}
primitiv_Node *safe_primitiv_node_func_reshape(
    const primitiv_Node *x,
    const primitiv_Shape *new_shape,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_reshape(x, new_shape), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_reshape(
    const primitiv_Tensor *x,
    const primitiv_Shape *new_shape) {
  return to_c_from_value(
      primitiv::functions::reshape(*to_cc(x), *to_cc(new_shape)));
}
primitiv_Tensor *safe_primitiv_tensor_func_reshape(
    const primitiv_Tensor *x,
    const primitiv_Shape *new_shape,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_reshape(x, new_shape), status, nullptr);
}

IMPL_UNARY_FUNC(flatten);
IMPL_UNARY_FUNC(transpose);

primitiv_Node *primitiv_node_func_matmul(const primitiv_Node *a,
                                         const primitiv_Node *b) {
  return to_c_from_value(primitiv::functions::matmul(*to_cc(a), *to_cc(b)));
}
primitiv_Node *safe_primitiv_node_func_matmul(const primitiv_Node *a,
                                              const primitiv_Node *b,
                                              primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_matmul(a, b), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_matmul(const primitiv_Tensor *a,
                                             const primitiv_Tensor *b) {
  return to_c_from_value(primitiv::functions::matmul(*to_cc(a), *to_cc(b)));
}
primitiv_Tensor *safe_primitiv_tensor_func_matmul(const primitiv_Tensor *a,
                                                  const primitiv_Tensor *b,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_matmul(a, b), status, nullptr);
}

IMPL_UNARY_FUNC(sqrt);
IMPL_UNARY_FUNC(exp);
IMPL_UNARY_FUNC(log);
IMPL_UNARY_FUNC(tanh);
IMPL_UNARY_FUNC(sigmoid);
IMPL_UNARY_FUNC(softplus);
IMPL_UNARY_FUNC(sin);
IMPL_UNARY_FUNC(cos);
IMPL_UNARY_FUNC(tan);
IMPL_UNARY_FUNC(relu);
IMPL_UNARY_FUNC(lrelu);

primitiv_Node *primitiv_node_func_prelu(const primitiv_Node *x, float a) {
  return to_c_from_value(primitiv::functions::prelu(*to_cc(x), a));
}
primitiv_Node *safe_primitiv_node_func_prelu(const primitiv_Node *x,
                                             float a,
                                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_prelu(x, a), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_prelu(const primitiv_Tensor *x, float a) {
  return to_c_from_value(primitiv::functions::prelu(*to_cc(x), a));
}
primitiv_Tensor *safe_primitiv_tensor_func_prelu(const primitiv_Tensor *x,
                                                 float a,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_prelu(x, a), status, nullptr);
}

primitiv_Node *primitiv_node_func_elu(const primitiv_Node *x, float a) {
  return to_c_from_value(primitiv::functions::elu(*to_cc(x), a));
}
primitiv_Node *safe_primitiv_node_func_elu(const primitiv_Node *x,
                                           float a,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_elu(x, a), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_elu(const primitiv_Tensor *x, float a) {
  return to_c_from_value(primitiv::functions::elu(*to_cc(x), a));
}
primitiv_Tensor *safe_primitiv_tensor_func_elu(const primitiv_Tensor *x,
                                               float a,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_elu(x, a), status, nullptr);
}

primitiv_Node *primitiv_node_func_sum(const primitiv_Node *x, uint32_t dim) {
  return to_c_from_value(primitiv::functions::sum(*to_cc(x), dim));
}
primitiv_Node *safe_primitiv_node_func_sum(const primitiv_Node *x,
                                           uint32_t dim,
                                           primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_sum(x, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_sum(const primitiv_Tensor *x,
                                          uint32_t dim) {
  return to_c_from_value(primitiv::functions::sum(*to_cc(x), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_sum(const primitiv_Tensor *x,
                                               uint32_t dim,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_sum(x, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_broadcast(const primitiv_Node *x,
                                            uint32_t dim,
                                            uint32_t size) {
  return to_c_from_value(primitiv::functions::broadcast(*to_cc(x), dim, size));
}
primitiv_Node *safe_primitiv_node_func_broadcast(const primitiv_Node *x,
                                                 uint32_t dim,
                                                 uint32_t size,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_broadcast(x, dim, size), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_broadcast(const primitiv_Tensor *x,
                                                uint32_t dim,
                                                uint32_t size) {
  return to_c_from_value(primitiv::functions::broadcast(*to_cc(x), dim, size));
}
primitiv_Tensor *safe_primitiv_tensor_func_broadcast(const primitiv_Tensor *x,
                                                     uint32_t dim,
                                                     uint32_t size,
                                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_broadcast(x, dim, size), status, nullptr);
}

primitiv_Node *primitiv_node_func_logsumexp(const primitiv_Node *x,
                                            uint32_t dim) {
  return to_c_from_value(primitiv::functions::logsumexp(*to_cc(x), dim));
}
primitiv_Node *safe_primitiv_node_func_logsumexp(const primitiv_Node *x,
                                                 uint32_t dim,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_logsumexp(x, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_logsumexp(const primitiv_Tensor *x,
                                                uint32_t dim) {
  return to_c_from_value(primitiv::functions::logsumexp(*to_cc(x), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_logsumexp(const primitiv_Tensor *x,
                                                     uint32_t dim,
                                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_logsumexp(x, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_log_softmax(const primitiv_Node *x,
                                              uint32_t dim) {
  return to_c_from_value(primitiv::functions::log_softmax(*to_cc(x), dim));
}
primitiv_Node *safe_primitiv_node_func_log_softmax(const primitiv_Node *x,
                                                   uint32_t dim,
                                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_log_softmax(x, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_log_softmax(const primitiv_Tensor *x,
                                                  uint32_t dim) {
  return to_c_from_value(primitiv::functions::log_softmax(*to_cc(x), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_log_softmax(
    const primitiv_Tensor *x, uint32_t dim, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_log_softmax(x, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_softmax(const primitiv_Node *x,
                                          uint32_t dim) {
  return to_c_from_value(primitiv::functions::softmax(*to_cc(x), dim));
}
primitiv_Node *safe_primitiv_node_func_softmax(const primitiv_Node *x,
                                               uint32_t dim,
                                               primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_softmax(x, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_softmax(const primitiv_Tensor *x,
                                              uint32_t dim) {
  return to_c_from_value(primitiv::functions::softmax(*to_cc(x), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_softmax(const primitiv_Tensor *x,
                                                   uint32_t dim,
                                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_softmax(x, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_softmax_cross_entropy(const primitiv_Node *x,
                                                        const primitiv_Node *t,
                                                        uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::softmax_cross_entropy(*to_cc(x), *to_cc(t), dim));
}
primitiv_Node *safe_primitiv_node_func_softmax_cross_entropy(
    const primitiv_Node *x,
    const primitiv_Node *t,
    uint32_t dim,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_node_func_softmax_cross_entropy(x, t, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_softmax_cross_entropy(
    const primitiv_Tensor *x,
    const primitiv_Tensor *t,
    uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::softmax_cross_entropy(*to_cc(x), *to_cc(t), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_softmax_cross_entropy(
    const primitiv_Tensor *x,
    const primitiv_Tensor *t,
    uint32_t dim,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_softmax_cross_entropy(x, t, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_softmax_cross_entropy_with_array(
    const primitiv_Node *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cc(x), std::vector<uint32_t>(ids, ids + n), dim));
}
primitiv_Node *safe_primitiv_node_func_softmax_cross_entropy_with_array(
    const primitiv_Node *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_node_func_softmax_cross_entropy_with_array(x, ids, n, dim),
      status,
      nullptr);
}
primitiv_Tensor *primitiv_tensor_func_softmax_cross_entropy_with_array(
    const primitiv_Tensor *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim) {
  return to_c_from_value(
      primitiv::functions::softmax_cross_entropy(
          *to_cc(x), std::vector<uint32_t>(ids, ids + n), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_softmax_cross_entropy_with_array(
    const primitiv_Tensor *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_softmax_cross_entropy_with_array(x, ids, n, dim),
      status,
      nullptr);
}

IMPL_UNARY_FUNC(stop_gradient);

primitiv_Node *primitiv_node_func_batch_sum(const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::batch::sum(*to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_batch_sum(const primitiv_Node *x,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_batch_sum(x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_batch_sum(const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::batch::sum(*to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_batch_sum(const primitiv_Tensor *x,
                                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_batch_sum(x), status, nullptr);
}

primitiv_Node *primitiv_node_func_constant(const primitiv_Shape *shape,
                                           float k,
                                           primitiv_Device *dev,
                                           primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::constant_node(
      *to_cc(shape), k, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_constant(const primitiv_Shape *shape,
                                                float k,
                                                primitiv_Device *dev,
                                                primitiv_Graph *g,
                                                primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_node_func_constant(shape, k, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_constant(const primitiv_Shape *shape,
                                               float k,
                                               primitiv_Device *dev) {
  return to_c_from_value(
      primitiv::functions::constant_tensor(*to_cc(shape), k, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_constant(const primitiv_Shape *shape,
                                                    float k,
                                                    primitiv_Device *dev,
                                                    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_constant(shape, k, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_identity(uint32_t size,
                                           primitiv_Device *dev,
                                           primitiv_Graph *g) {
  return to_c_from_value(
      primitiv::functions::identity_node(size, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_identity(uint32_t size,
                                                primitiv_Device *dev,
                                                primitiv_Graph *g,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_identity(size, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_identity(uint32_t size,
                                               primitiv_Device *dev) {
  return to_c_from_value(
      primitiv::functions::identity_tensor(size, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_identity(uint32_t size,
                                                    primitiv_Device *dev,
                                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_identity(size, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_random_bernoulli(const primitiv_Shape *shape,
                                                   float p,
                                                   primitiv_Device *dev,
                                                   primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::random::bernoulli_node(
      *to_cc(shape), p, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_node_func_random_bernoulli(shape, p, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::random::bernoulli_tensor(
      *to_cc(shape), p, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_random_bernoulli(shape, p, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_random_uniform(const primitiv_Shape *shape,
                                                 float lower,
                                                 float upper,
                                                 primitiv_Device *dev,
                                                 primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::random::uniform_node(
      *to_cc(shape), lower, upper, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_random_uniform(
      shape, lower, upper, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::random::uniform_tensor(
      *to_cc(shape), lower, upper, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_random_uniform(
      shape, lower, upper, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_random_normal(const primitiv_Shape *shape,
                                                float mean,
                                                float sd,
                                                primitiv_Device *dev,
                                                primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::random::normal_node(
      *to_cc(shape), mean, sd, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_random_normal(
      shape, mean, sd, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::random::normal_tensor(
      *to_cc(shape), mean, sd, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_random_normal(
      shape, mean, sd, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_random_log_normal(const primitiv_Shape *shape,
                                                    float mean,
                                                    float sd,
                                                    primitiv_Device *dev,
                                                    primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::random::log_normal_node(
      *to_cc(shape), mean, sd, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_random_log_normal(
      shape, mean, sd, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::random::log_normal_tensor(
      *to_cc(shape), mean, sd, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_random_normal(
      shape, mean, sd, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_random_gumbel(const primitiv_Shape *shape,
                                                float mu,
                                                float beta,
                                                primitiv_Device *dev,
                                                primitiv_Graph *g) {
  return to_c_from_value(primitiv::functions::random::gumbel_node(
      *to_cc(shape), mu, beta, to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_random_gumbel(
      shape, mu, beta, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev) {
  return to_c_from_value(primitiv::functions::random::gumbel_tensor(
      *to_cc(shape), mu, beta, to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_random_gumbel(
      shape, mu, beta, dev), status, nullptr);
}

IMPL_BINARY_OPERATOR(pow);

primitiv_Node *primitiv_node_func_pown(const primitiv_Node *x, uint32_t k) {
  return to_c_from_value(primitiv::functions::pown(*to_cc(x), k));
}
primitiv_Node *safe_primitiv_node_func_pown(const primitiv_Node *x,
                                            uint32_t k,
                                            primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_pown(x, k), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_pown(const primitiv_Tensor *x,
                                           uint32_t k) {
  return to_c_from_value(primitiv::functions::pown(*to_cc(x), k));
}
primitiv_Tensor *safe_primitiv_tensor_func_pown(const primitiv_Tensor *x,
                                                uint32_t k,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_pown(x, k), status, nullptr);
}

IMPL_UNARY_FUNC(selu);

primitiv_Node *primitiv_node_func_sum_nodes(const primitiv_Node *const *xs,
                                            size_t n) {
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  return to_c_from_value(
      primitiv::functions::sum(std::vector<const Node*>(_xs, _xs + n)));
}
primitiv_Node *safe_primitiv_node_func_sum_nodes(const primitiv_Node *const *xs,
                                                 size_t n,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_sum_nodes(xs, n), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_sum_tensors(
    const primitiv_Tensor *const *xs,
    size_t n) {
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  return to_c_from_value(
      primitiv::functions::sum(std::vector<const Tensor*>(_xs, _xs + n)));
}
primitiv_Tensor *safe_primitiv_tensor_func_sum_tensors(
    const primitiv_Tensor *const *xs,
    size_t n,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_sum_tensors(xs, n), status, nullptr);
}

primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x, uint32_t dim) {
  return to_c_from_value(primitiv::functions::mean(*to_cc(x), dim));
}
primitiv_Node *safe_primitiv_node_func_mean(const primitiv_Node *x,
                                            uint32_t dim,
                                            primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_mean(x, dim), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x,
                                           uint32_t dim) {
  return to_c_from_value(primitiv::functions::mean(*to_cc(x), dim));
}
primitiv_Tensor *safe_primitiv_tensor_func_mean(const primitiv_Tensor *x,
                                                uint32_t dim,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_mean(x, dim), status, nullptr);
}

primitiv_Node *primitiv_node_func_mean_nodes(const primitiv_Node *const *xs,
                                             size_t n) {
  const Node *const *_xs = reinterpret_cast<const Node *const *>(xs);
  return to_c_from_value(
      primitiv::functions::mean(std::vector<const Node*>(_xs, _xs + n)));
}
primitiv_Node *safe_primitiv_node_func_mean_nodes(
    const primitiv_Node *const *xs,
    size_t n,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_mean_nodes(xs, n), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_mean_tensors(
    const primitiv_Tensor *const *xs,
    size_t n) {
  const Tensor *const *_xs = reinterpret_cast<const Tensor *const *>(xs);
  return to_c_from_value(
      primitiv::functions::mean(std::vector<const Tensor*>(_xs, _xs + n)));
}
primitiv_Tensor *safe_primitiv_tensor_func_mean_tensors(
    const primitiv_Tensor *const *xs,
    size_t n,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_mean_tensors(xs, n), status, nullptr);
}

primitiv_Node *primitiv_node_func_batch_mean(const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::batch::mean(*to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_batch_mean(const primitiv_Node *x,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_batch_mean(x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_batch_mean(const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::batch::mean(*to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_batch_mean(const primitiv_Tensor *x,
                                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_batch_mean(x), status, nullptr);
}

primitiv_Node *primitiv_node_func_batch_normalize(const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::batch::normalize(*to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_batch_normalize(const primitiv_Node *x,
                                                  primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_batch_normalize(x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_batch_normalize(
    const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::batch::normalize(*to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_batch_normalize(
    const primitiv_Tensor *x,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_batch_normalize(x), status, nullptr);
}

primitiv_Node *primitiv_node_func_zeros(const primitiv_Shape *shape,
                                        primitiv_Device *dev,
                                        primitiv_Graph *g) {
  return to_c_from_value(
      primitiv::functions::zeros_node(*to_cc(shape), to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_zeros(const primitiv_Shape *shape,
                                             primitiv_Device *dev,
                                             primitiv_Graph *g,
                                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_zeros(shape, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_zeros(const primitiv_Shape *shape,
                                            primitiv_Device *dev) {
  return to_c_from_value(
      primitiv::functions::zeros_tensor(*to_cc(shape), to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_zeros(const primitiv_Shape *shape,
                                                 primitiv_Device *dev,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_zeros(shape, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_ones(const primitiv_Shape *shape,
                                        primitiv_Device *dev,
                                        primitiv_Graph *g) {
  return to_c_from_value(
      primitiv::functions::ones_node(*to_cc(shape), to_cc(dev), to_cc(g)));
}
primitiv_Node *safe_primitiv_node_func_ones(const primitiv_Shape *shape,
                                             primitiv_Device *dev,
                                             primitiv_Graph *g,
                                             primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_ones(shape, dev, g), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_ones(const primitiv_Shape *shape,
                                            primitiv_Device *dev) {
  return to_c_from_value(
      primitiv::functions::ones_tensor(*to_cc(shape), to_cc(dev)));
}
primitiv_Tensor *safe_primitiv_tensor_func_ones(const primitiv_Shape *shape,
                                                 primitiv_Device *dev,
                                                 primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_ones(shape, dev), status, nullptr);
}

primitiv_Node *primitiv_node_func_dropout(const primitiv_Node *x,
                                          float rate,
                                          bool enabled) {
  return to_c_from_value(
      primitiv::functions::dropout(*to_cc(x), rate, enabled));
}
primitiv_Node *safe_primitiv_node_func_dropout(
    const primitiv_Node *x, float rate, bool enabled, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_dropout(x, rate, enabled), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_dropout(const primitiv_Tensor *x,
                                              float rate,
                                              bool enabled) {
  return to_c_from_value(
      primitiv::functions::dropout(*to_cc(x), rate, enabled));
}
primitiv_Tensor *safe_primitiv_tensor_func_dropout(const primitiv_Tensor *x,
                                                   float rate,
                                                   bool enabled,
                                                   primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_dropout(x, rate, enabled), status, nullptr);
}

}  // end extern "C"
