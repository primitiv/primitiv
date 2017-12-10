/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#include <primitiv/functions.h>

#include <vector>

#include "primitiv_c/internal.h"
#include "primitiv_c/functions.h"

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

}  // end extern "C"
