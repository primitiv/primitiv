#ifndef PRIMITIV_C_OPERATORS_H_
#define PRIMITIV_C_OPERATORS_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/graph.h"
#include "primitiv_c/parameter.h"
#include "primitiv_c/shape.h"
#include "primitiv_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Node *primitiv_func_node_add_const(const primitiv_Node *x, float k);

primitiv_Node *primitiv_func_const_add_node(float k, const primitiv_Node *x);

primitiv_Node *primitiv_func_node_add_node(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Node *primitiv_func_node_sub_const(const primitiv_Node *x, float k);

primitiv_Node *primitiv_func_const_sub_node(float k, const primitiv_Node *x);

primitiv_Node *primitiv_func_node_sub_node(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Node *primitiv_func_node_mul_const(const primitiv_Node *x, float k);

primitiv_Node *primitiv_func_const_mul_node(float k, const primitiv_Node *x);

primitiv_Node *primitiv_func_node_mul_node(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Node *primitiv_func_node_div_const(const primitiv_Node *x, float k);

primitiv_Node *primitiv_func_const_div_node(float k, const primitiv_Node *x);

primitiv_Node *primitiv_func_node_div_node(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Tensor *primitiv_func_tensor_add_const(const primitiv_Tensor *x, float k);

primitiv_Tensor *primitiv_func_const_add_tensor(float k, const primitiv_Tensor *x);

primitiv_Tensor *primitiv_func_tensor_add_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Tensor *primitiv_func_tensor_sub_const(const primitiv_Tensor *x, float k);

primitiv_Tensor *primitiv_func_const_sub_tensor(float k, const primitiv_Tensor *x);

primitiv_Tensor *primitiv_func_tensor_sub_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Tensor *primitiv_func_tensor_mul_const(const primitiv_Tensor *x, float k);

primitiv_Tensor *primitiv_func_const_mul_tensor(float k, const primitiv_Tensor *x);

primitiv_Tensor *primitiv_func_tensor_mul_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Tensor *primitiv_func_tensor_div_const(const primitiv_Tensor *x, float k);

primitiv_Tensor *primitiv_func_const_div_tensor(float k, const primitiv_Tensor *x);

primitiv_Tensor *primitiv_func_tensor_div_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x, uint32_t dim);

primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x, uint32_t dim);

primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x, uint32_t dim);

primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x, uint32_t dim);

primitiv_Node *primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device);

primitiv_Tensor *primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device);

primitiv_Node *primitiv_node_func_parameter(primitiv_Parameter *param);

primitiv_Tensor *primitiv_tensor_func_parameter(primitiv_Parameter *param);

primitiv_Node *primitiv_node_func_tanh(const primitiv_Node *x);

primitiv_Tensor *primitiv_tensor_func_tanh(const primitiv_Tensor *x);

primitiv_Node *primitiv_node_func_matmul(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Tensor *primitiv_tensor_func_matmul(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Node *primitiv_node_func_batch_mean(const primitiv_Node *x);

primitiv_Tensor *primitiv_tensor_func_batch_mean(const primitiv_Tensor *x);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPERATORS_H_
