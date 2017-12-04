#ifndef PRIMITIV_C_OPERATORS_H_
#define PRIMITIV_C_OPERATORS_H_

#include "primitiv_c/define.h"
#include "primitiv_c/shape.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Node *primitiv_node_op_mean(const primitiv_Node *x, uint32_t dim);

primitiv_Tensor *primitiv_tensor_op_mean(const primitiv_Tensor *x, uint32_t dim);

primitiv_Node *primitiv_node_op_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device);

primitiv_Tensor *primitiv_tensor_op_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device);

primitiv_Node *primitiv_node_op_parameter(primitiv_Parameter *param);

primitiv_Tensor *primitiv_tensor_op_parameter(primitiv_Parameter *param);

primitiv_Node *primitiv_node_op_tanh(const primitiv_Node *x);

primitiv_Tensor *primitiv_tensor_op_tanh(const primitiv_Tensor *x);

primitiv_Node *primitiv_node_op_matmul(const primitiv_Node *a, const primitiv_Node *b);

primitiv_Tensor *primitiv_tensor_op_matmul(const primitiv_Tensor *a, const primitiv_Tensor *b);

primitiv_Node *primitiv_node_op_batch_mean(const primitiv_Node *x);

primitiv_Tensor *primitiv_tensor_op_batch_mean(const primitiv_Tensor *x);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // PRIMITIV_C_OPERATORS_H_
