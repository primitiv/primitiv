/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#ifndef PRIMITIV_C_FUNCTIONS_H_
#define PRIMITIV_C_FUNCTIONS_H_

#include <primitiv/c/define.h>
#include <primitiv/c/device.h>
#include <primitiv/c/graph.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/shape.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

#define _CONCAT_EVAL(A, B) A ## B
#define _CONCAT(A, B) _CONCAT_EVAL(A, B)
#define _NODE_FUN(name) _CONCAT(primitiv_node_func_, name)
#define _S_NODE_FUN(name) _CONCAT(safe_primitiv_node_func_, name)
#define _TENSOR_FUN(name) _CONCAT(primitiv_tensor_func_, name)
#define _S_TENSOR_FUN(name) _CONCAT(safe_primitiv_tensor_func_, name)
#define _NODE_OP(A, B) _NODE_FUN(_CONCAT(A, B))
#define _S_NODE_OP(A, B) _S_NODE_FUN(_CONCAT(A, B))
#define _TENSOR_OP(A, B) _TENSOR_FUN(_CONCAT(A, B))
#define _S_TENSOR_OP(A, B) _S_TENSOR_FUN(_CONCAT(A, B))

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_UNARY_FUNC(name) \
CAPI extern primitiv_Node *_NODE_FUN(name)(const primitiv_Node *x); \
CAPI extern primitiv_Node *_S_NODE_FUN(name)( \
    const primitiv_Node *x, primitiv_Status *status); \
CAPI extern primitiv_Tensor *_TENSOR_FUN(name)( \
    const primitiv_Tensor *x); \
CAPI extern primitiv_Tensor *_S_TENSOR_FUN(name)( \
    const primitiv_Tensor *x, primitiv_Status *status)

#define DEFINE_BINARY_OPERATOR(op_name) \
CAPI extern primitiv_Node *_NODE_OP(op_name, _node_const)( \
    const primitiv_Node *x, float k); \
CAPI extern primitiv_Node *_S_NODE_OP(op_name, _node_const)( \
    const primitiv_Node *x, float k, primitiv_Status *status); \
CAPI extern primitiv_Node *_NODE_OP(op_name, _const_node)( \
    float k, const primitiv_Node *x); \
CAPI extern primitiv_Node *_S_NODE_OP(op_name, _const_node)( \
    float k, const primitiv_Node *x, primitiv_Status *status); \
CAPI extern primitiv_Node *_NODE_OP(op_name, _node_node)( \
    const primitiv_Node *a, const primitiv_Node *b); \
CAPI extern primitiv_Node *_S_NODE_OP(op_name, _node_node)( \
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status); \
CAPI extern primitiv_Tensor *_TENSOR_OP(op_name, _tensor_const)( \
    const primitiv_Tensor *x, float k); \
CAPI extern primitiv_Tensor *_S_TENSOR_OP(op_name, _tensor_const)( \
    const primitiv_Tensor *x, float k, primitiv_Status *status); \
CAPI extern primitiv_Tensor *_TENSOR_OP(op_name, _const_tensor)( \
    float k, const primitiv_Tensor *x); \
CAPI extern primitiv_Tensor *_S_TENSOR_OP(op_name, _const_tensor)( \
    float k, const primitiv_Tensor *x, primitiv_Status *status); \
CAPI extern primitiv_Tensor *_TENSOR_OP(op_name, _tensor_tensor)( \
    const primitiv_Tensor *a, const primitiv_Tensor *b); \
CAPI extern primitiv_Tensor *_S_TENSOR_OP(op_name, _tensor_tensor)( \
    const primitiv_Tensor *a, \
    const primitiv_Tensor *b, \
    primitiv_Status *status)

DEFINE_UNARY_FUNC(positive);
DEFINE_UNARY_FUNC(negative);
DEFINE_BINARY_OPERATOR(add);
DEFINE_BINARY_OPERATOR(subtract);
DEFINE_BINARY_OPERATOR(multiply);
DEFINE_BINARY_OPERATOR(divide);

CAPI extern primitiv_Node *primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_parameter(
    primitiv_Parameter *param, primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_parameter(
    primitiv_Parameter *param, primitiv_Graph *g, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_parameter(
    primitiv_Parameter *param);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_parameter(
    primitiv_Parameter *param, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_copy(const primitiv_Node *x,
                                                   primitiv_Device *dev);
CAPI extern primitiv_Node *safe_primitiv_node_func_copy(
    const primitiv_Node *x, primitiv_Device *dev, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_copy(const primitiv_Tensor *x,
                                                       primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_copy(
    const primitiv_Tensor *x, primitiv_Device *dev, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_pick(const primitiv_Node *x,
                                                   const uint32_t *ids,
                                                   size_t n,
                                                   uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_pick(
    const primitiv_Node *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_pick(const primitiv_Tensor *x,
                                                       const uint32_t *ids,
                                                       size_t n,
                                                       uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_pick(
    const primitiv_Tensor *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_slice(const primitiv_Node *x,
                                                    uint32_t dim,
                                                    uint32_t lower,
                                                    uint32_t upper);
CAPI extern primitiv_Node *safe_primitiv_node_func_slice(
    const primitiv_Node *x,
    uint32_t dim,
    uint32_t lower,
    uint32_t upper,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_slice(
    const primitiv_Tensor *x,
    uint32_t dim,
    uint32_t lower,
    uint32_t upper);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_slice(
    const primitiv_Tensor *x,
    uint32_t dim,
    uint32_t lower,
    uint32_t upper,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_concat(
    const primitiv_Node *const *xs,
    size_t n,
    uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_concat(
    const primitiv_Node *const *xs,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_concat(
    const primitiv_Tensor *const *xs,
    size_t n,
    uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_concat(
    const primitiv_Tensor *const *xs,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_reshape(
    const primitiv_Node *x,
    const primitiv_Shape *new_shape);
CAPI extern primitiv_Node *safe_primitiv_node_func_reshape(
    const primitiv_Node *x,
    const primitiv_Shape *new_shape,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_reshape(
    const primitiv_Tensor *x,
    const primitiv_Shape *new_shape);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_reshape(
    const primitiv_Tensor *x,
    const primitiv_Shape *new_shape,
    primitiv_Status *status);

DEFINE_UNARY_FUNC(flatten);
DEFINE_UNARY_FUNC(transpose);

CAPI extern primitiv_Node *primitiv_node_func_matmul(const primitiv_Node *a,
                                                     const primitiv_Node *b);
CAPI extern primitiv_Node *safe_primitiv_node_func_matmul(
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_matmul(
    const primitiv_Tensor *a, const primitiv_Tensor *b);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_matmul(
    const primitiv_Tensor *a,
    const primitiv_Tensor *b,
    primitiv_Status *status);

DEFINE_UNARY_FUNC(sqrt);
DEFINE_UNARY_FUNC(exp);
DEFINE_UNARY_FUNC(log);
DEFINE_UNARY_FUNC(tanh);
DEFINE_UNARY_FUNC(sigmoid);
DEFINE_UNARY_FUNC(softplus);
DEFINE_UNARY_FUNC(sin);
DEFINE_UNARY_FUNC(cos);
DEFINE_UNARY_FUNC(tan);
DEFINE_UNARY_FUNC(relu);
DEFINE_UNARY_FUNC(lrelu);

CAPI extern primitiv_Node *primitiv_node_func_prelu(const primitiv_Node *x,
                                                    float a);
CAPI extern primitiv_Node *safe_primitiv_node_func_prelu(
    const primitiv_Node *x, float a, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_prelu(
    const primitiv_Tensor *x, float a);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_prelu(
    const primitiv_Tensor *x, float a, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_elu(const primitiv_Node *x,
                                                  float a);
CAPI extern primitiv_Node *safe_primitiv_node_func_elu(
    const primitiv_Node *x, float a, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_elu(
    const primitiv_Tensor *x, float a);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_elu(
    const primitiv_Tensor *x, float a, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_sum(const primitiv_Node *x,
                                                  uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_sum(
    const primitiv_Node *x, uint32_t dim, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_sum(
    const primitiv_Tensor *x, uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_sum(
    const primitiv_Tensor *x, uint32_t dim, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_broadcast(const primitiv_Node *x,
                                                        uint32_t dim,
                                                        uint32_t size);
CAPI extern primitiv_Node *safe_primitiv_node_func_broadcast(
    const primitiv_Node *x,
    uint32_t dim,
    uint32_t size,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_broadcast(
    const primitiv_Tensor *x, uint32_t dim, uint32_t size);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_broadcast(
    const primitiv_Tensor *x,
    uint32_t dim,
    uint32_t size,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_logsumexp(const primitiv_Node *x,
                                                        uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_logsumexp(
    const primitiv_Node *x, uint32_t dim, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_logsumexp(
    const primitiv_Tensor *x, uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_logsumexp(
    const primitiv_Tensor *x, uint32_t dim, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_log_softmax(
    const primitiv_Node *x, uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_log_softmax(
    const primitiv_Node *x, uint32_t dim, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_log_softmax(
    const primitiv_Tensor *x, uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_log_softmax(
    const primitiv_Tensor *x, uint32_t dim, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_softmax(const primitiv_Node *x,
                                                      uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_softmax(
    const primitiv_Node *x, uint32_t dim, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_softmax(
    const primitiv_Tensor *x, uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_softmax(
    const primitiv_Tensor *x, uint32_t dim, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_softmax_cross_entropy(
    const primitiv_Node *x,
    const primitiv_Node *t,
    uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_softmax_cross_entropy(
    const primitiv_Node *x,
    const primitiv_Node *t,
    uint32_t dim,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_softmax_cross_entropy(
    const primitiv_Tensor *x,
    const primitiv_Tensor *t,
    uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_softmax_cross_entropy(
    const primitiv_Tensor *x,
    const primitiv_Tensor *t,
    uint32_t dim,
    primitiv_Status *status);

CAPI extern primitiv_Node
*primitiv_node_func_softmax_cross_entropy_with_array(
    const primitiv_Node *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim);
CAPI extern primitiv_Node
*safe_primitiv_node_func_softmax_cross_entropy_with_array(
    const primitiv_Node *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);
CAPI extern primitiv_Tensor
*primitiv_tensor_func_softmax_cross_entropy_with_array(
    const primitiv_Tensor *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim);
CAPI extern primitiv_Tensor
*safe_primitiv_tensor_func_softmax_cross_entropy_with_array(
    const primitiv_Tensor *x,
    const uint32_t *ids,
    size_t n,
    uint32_t dim,
    primitiv_Status *status);

DEFINE_UNARY_FUNC(stop_gradient);

CAPI extern primitiv_Node *primitiv_node_func_batch_sum(const primitiv_Node *x);
CAPI extern primitiv_Node *safe_primitiv_node_func_batch_sum(
    const primitiv_Node *x, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_batch_sum(
    const primitiv_Tensor *x);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_batch_sum(
    const primitiv_Tensor *x, primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_constant(
    const primitiv_Shape *shape,
    float k,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_constant(
    const primitiv_Shape *shape,
    float k,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_constant(
    const primitiv_Shape *shape,
    float k,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_constant(
    const primitiv_Shape *shape,
    float k,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_identity(
    uint32_t size,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_identity(
    uint32_t size,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_identity(
    uint32_t size,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_identity(
    uint32_t size,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_random_bernoulli(
    const primitiv_Shape *shape,
    float p,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_random_uniform(
    const primitiv_Shape *shape,
    float lower,
    float upper,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_random_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_random_log_normal(
    const primitiv_Shape *shape,
    float mean,
    float sd,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev,
    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_random_gumbel(
    const primitiv_Shape *shape,
    float mu,
    float beta,
    primitiv_Device *dev,
    primitiv_Status *status);

DEFINE_BINARY_OPERATOR(pow);

CAPI extern primitiv_Node *primitiv_node_func_ipow(const primitiv_Node *x,
                                                   uint32_t k);
CAPI extern primitiv_Node *safe_primitiv_node_func_ipow(
    const primitiv_Node *x,
    uint32_t k,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_ipow(const primitiv_Tensor *x,
                                                       uint32_t k);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_ipow(
    const primitiv_Tensor *x,
    uint32_t k,
    primitiv_Status *status);

DEFINE_UNARY_FUNC(selu);

CAPI extern primitiv_Node *primitiv_node_func_sum_nodes(
    const primitiv_Node *const *xs,
    size_t n);
CAPI extern primitiv_Node *safe_primitiv_node_func_sum_nodes(
    const primitiv_Node *const *xs,
    size_t n,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_sum_tensors(
    const primitiv_Tensor *const *xs,
    size_t n);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_sum_tensors(
    const primitiv_Tensor *const *xs,
    size_t n,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x,
                                                   uint32_t dim);
CAPI extern primitiv_Node *safe_primitiv_node_func_mean(
    const primitiv_Node *x,
    uint32_t dim,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x,
                                                       uint32_t dim);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_mean(
    const primitiv_Tensor *x,
    uint32_t dim,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_mean_nodes(
    const primitiv_Node *const *xs,
    size_t n);
CAPI extern primitiv_Node *safe_primitiv_node_func_mean_nodes(
    const primitiv_Node *const *xs,
    size_t n,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_mean_tensors(
    const primitiv_Tensor *const *xs,
    size_t n);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_mean_tensor(
    const primitiv_Tensor *const *xs,
    size_t n,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_batch_mean(
    const primitiv_Node *x);
CAPI extern primitiv_Node *safe_primitiv_node_func_batch_mean(
    const primitiv_Node *x,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_batch_mean(
    const primitiv_Tensor *x);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_batch_mean(
    const primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_batch_normalize(
    const primitiv_Node *x);
CAPI extern primitiv_Node *safe_primitiv_node_func_batch_normalize(
    const primitiv_Node *x,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_batch_normalize(
    const primitiv_Tensor *x);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_batch_normalize(
    const primitiv_Tensor *x,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_zeros(const primitiv_Shape *shape,
                                                    primitiv_Device *dev,
                                                    primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_zeros(
    const primitiv_Shape *shape,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_zeros(
    const primitiv_Shape *shape,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_zeros(
    const primitiv_Shape *shape,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_ones(const primitiv_Shape *shape,
                                                   primitiv_Device *dev,
                                                   primitiv_Graph *g);
CAPI extern primitiv_Node *safe_primitiv_node_func_ones(
    const primitiv_Shape *shape,
    primitiv_Device *dev,
    primitiv_Graph *g,
    primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_ones(
    const primitiv_Shape *shape,
    primitiv_Device *dev);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_ones(
    const primitiv_Shape *shape,
    primitiv_Device *dev,
    primitiv_Status *status);

CAPI extern primitiv_Node *primitiv_node_func_dropout(const primitiv_Node *x,
                                                      float rate,
                                                      bool enabled);
CAPI extern primitiv_Node *safe_primitiv_node_func_dropout(
    const primitiv_Node *x, float rate, bool enabled, primitiv_Status *status);
CAPI extern primitiv_Tensor *primitiv_tensor_func_dropout(
    const primitiv_Tensor *x, float rate, bool enabled);
CAPI extern primitiv_Tensor *safe_primitiv_tensor_func_dropout(
    const primitiv_Tensor *x,
    float rate,
    bool enabled,
    primitiv_Status *status);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_FUNCTIONS_H_
