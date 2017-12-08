#include "primitiv_c/internal.h"
#include "primitiv_c/functions.h"

#include <vector>

#include <primitiv/functions.h>

using primitiv::Node;
using primitiv::Tensor;

extern "C" {

primitiv_Node *primitiv_node_func_node_add_const(const primitiv_Node *x, float k) {
  Node y = *to_cc(x) + k;
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_const_add_node(float k, const primitiv_Node *x) {
  Node y = k + *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_add_node(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = *to_cc(a) + *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_sub_const(const primitiv_Node *x, float k) {
  Node y = *to_cc(x) - k;
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_const_sub_node(float k, const primitiv_Node *x) {
  Node y = k - *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_sub_node(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = *to_cc(a) - *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_mul_const(const primitiv_Node *x, float k) {
  Node y = *to_cc(x) * k;
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_const_mul_node(float k, const primitiv_Node *x) {
  Node y = k * *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_mul_node(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = *to_cc(a) * *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_div_const(const primitiv_Node *x, float k) {
  Node y = *to_cc(x) / k;
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_const_div_node(float k, const primitiv_Node *x) {
  Node y = k / *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_node_div_node(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = *to_cc(a) / *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_add_const(const primitiv_Tensor *x, float k) {
  Tensor y = *to_cc(x) + k;
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_const_add_tensor(float k, const primitiv_Tensor *x) {
  Tensor y = k + *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_add_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = *to_cc(a) + *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_sub_const(const primitiv_Tensor *x, float k) {
  Tensor y = *to_cc(x) - k;
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_const_sub_tensor(float k, const primitiv_Tensor *x) {
  Tensor y = k - *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_sub_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = *to_cc(a) - *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_mul_const(const primitiv_Tensor *x, float k) {
  Tensor y = *to_cc(x) * k;
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_const_mul_tensor(float k, const primitiv_Tensor *x) {
  Tensor y = k * *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_mul_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = *to_cc(a) * *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_div_const(const primitiv_Tensor *x, float k) {
  Tensor y = *to_cc(x) / k;
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_const_div_tensor(float k, const primitiv_Tensor *x) {
  Tensor y = k / *to_cc(x);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tensor_div_tensor(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = *to_cc(a) / *to_cc(b);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x, uint32_t dim) {
  Node y = primitiv::functions::mean<Node>(*to_cc(x), dim);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x, uint32_t dim) {
  Tensor y = primitiv::functions::mean<Tensor>(*to_cc(x), dim);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  Node y = primitiv::functions::input<Node>(
      *to_cc(shape), std::vector<float>(data, data + n), *to_cc(device));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  Tensor y = primitiv::functions::input<Tensor>(
      *to_cc(shape), std::vector<float>(data, data + n), *to_cc(device));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_parameter(primitiv_Parameter *param) {
  Node y = primitiv::functions::parameter<Node>(*to_cc(param));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_parameter(primitiv_Parameter *param) {
  Tensor y = primitiv::functions::parameter<Tensor>(*to_cc(param));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_tanh(const primitiv_Node *x) {
  Node y = primitiv::functions::tanh<Node>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tanh(const primitiv_Tensor *x) {
  Tensor y = primitiv::functions::tanh<Tensor>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_matmul(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = primitiv::functions::matmul<Node>(*to_cc(a), *to_cc(b));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_matmul(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = primitiv::functions::matmul<Tensor>(*to_cc(a), *to_cc(b));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_batch_mean(const primitiv_Node *x) {
  Node y = primitiv::functions::batch::mean<Node>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_batch_mean(const primitiv_Tensor *x) {
  Tensor y = primitiv::functions::batch::mean<Tensor>(*to_cc(x));
  return to_c_from_value(y);
}

}  // end extern "C"
