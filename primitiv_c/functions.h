#ifndef PRIMITIV_C_OPERATORS_H_
#define PRIMITIV_C_OPERATORS_H_

#include "primitiv_c/define.h"
#include "primitiv_c/device.h"
#include "primitiv_c/graph.h"
#include "primitiv_c/parameter.h"
#include "primitiv_c/shape.h"
#include "primitiv_c/status.h"
#include "primitiv_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

primitiv_Node *primitiv_node_func_positive(const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_positive(const primitiv_Node *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_positive(const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_positive(const primitiv_Tensor *x, primitiv_Status *status);

primitiv_Node *primitiv_node_func_negative(const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_negative(const primitiv_Node *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_negative(const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_negative(const primitiv_Tensor *x, primitiv_Status *status);

primitiv_Node *primitiv_node_func_add_node_const(const primitiv_Node *x, float k);
primitiv_Node *safe_primitiv_node_func_add_node_const(const primitiv_Node *x, float k, primitiv_Status *status);
primitiv_Node *primitiv_node_func_add_const_node(float k, const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_add_const_node(float k, const primitiv_Node *x, primitiv_Status *status);
primitiv_Node *primitiv_node_func_add_node_node(const primitiv_Node *a, const primitiv_Node *b);
primitiv_Node *safe_primitiv_node_func_add_node_node(const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_add_tensor_const(const primitiv_Tensor *x, float k);
primitiv_Tensor *safe_primitiv_tensor_func_add_tensor_const(const primitiv_Tensor *x, float k, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_add_const_tensor(float k, const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_add_const_tensor(float k, const primitiv_Tensor *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_add_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);
primitiv_Tensor *safe_primitiv_tensor_func_add_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b, primitiv_Status *status);

primitiv_Node *primitiv_node_func_subtract_node_const(const primitiv_Node *x, float k);
primitiv_Node *safe_primitiv_node_func_subtract_node_const(const primitiv_Node *x, float k, primitiv_Status *status);
primitiv_Node *primitiv_node_func_subtract_const_node(float k, const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_subtract_const_node(float k, const primitiv_Node *x, primitiv_Status *status);
primitiv_Node *primitiv_node_func_subtract_node_node(const primitiv_Node *a, const primitiv_Node *b);
primitiv_Node *safe_primitiv_node_func_subtract_node_node(const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_subtract_tensor_const(const primitiv_Tensor *x, float k);
primitiv_Tensor *safe_primitiv_tensor_func_subtract_tensor_const(const primitiv_Tensor *x, float k, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_subtract_const_tensor(float k, const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_subtract_const_tensor(float k, const primitiv_Tensor *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_subtract_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);
primitiv_Tensor *safe_primitiv_tensor_func_subtract_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b, primitiv_Status *status);

primitiv_Node *primitiv_node_func_multiply_node_const(const primitiv_Node *x, float k);
primitiv_Node *safe_primitiv_node_func_multiply_node_const(const primitiv_Node *x, float k, primitiv_Status *status);
primitiv_Node *primitiv_node_func_multiply_const_node(float k, const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_multiply_const_node(float k, const primitiv_Node *x, primitiv_Status *status);
primitiv_Node *primitiv_node_func_multiply_node_node(const primitiv_Node *a, const primitiv_Node *b);
primitiv_Node *safe_primitiv_node_func_multiply_node_node(const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_multiply_tensor_const(const primitiv_Tensor *x, float k);
primitiv_Tensor *safe_primitiv_tensor_func_multiply_tensor_const(const primitiv_Tensor *x, float k, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_multiply_const_tensor(float k, const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_multiply_const_tensor(float k, const primitiv_Tensor *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_multiply_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);
primitiv_Tensor *safe_primitiv_tensor_func_multiply_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b, primitiv_Status *status);

primitiv_Node *primitiv_node_func_divide_node_const(const primitiv_Node *x, float k);
primitiv_Node *safe_primitiv_node_func_divide_node_const(const primitiv_Node *x, float k, primitiv_Status *status);
primitiv_Node *primitiv_node_func_divide_const_node(float k, const primitiv_Node *x);
primitiv_Node *safe_primitiv_node_func_divide_const_node(float k, const primitiv_Node *x, primitiv_Status *status);
primitiv_Node *primitiv_node_func_divide_node_node(const primitiv_Node *a, const primitiv_Node *b);
primitiv_Node *safe_primitiv_node_func_divide_node_node(const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_divide_tensor_const(const primitiv_Tensor *x, float k);
primitiv_Tensor *safe_primitiv_tensor_func_divide_tensor_const(const primitiv_Tensor *x, float k, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_divide_const_tensor(float k, const primitiv_Tensor *x);
primitiv_Tensor *safe_primitiv_tensor_func_divide_const_tensor(float k, const primitiv_Tensor *x, primitiv_Status *status);
primitiv_Tensor *primitiv_tensor_func_divide_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b);
primitiv_Tensor *safe_primitiv_tensor_func_divide_tensor_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b, primitiv_Status *status);

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
