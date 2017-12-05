#include "primitiv_c/internal.h"
#include "primitiv_c/operators.h"

#include <vector>

#include <primitiv/operators.h>

using primitiv::Node;
using primitiv::Tensor;

extern "C" {

primitiv_Node *primitiv_node_op_mean(const primitiv_Node *x, uint32_t dim) {
  Node y = primitiv::operators::mean<Node>(*to_cc(x), dim);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_mean(const primitiv_Tensor *x, uint32_t dim) {
  Tensor y = primitiv::operators::mean<Tensor>(*to_cc(x), dim);
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_op_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  Node y = primitiv::operators::input<Node>(
      *to_cc(shape), std::vector<float>(data, data + n), *to_cc(device));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_input(
    const primitiv_Shape *shape,
    const float *data,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  Tensor y = primitiv::operators::input<Tensor>(
      *to_cc(shape), std::vector<float>(data, data + n), *to_cc(device));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_op_parameter(primitiv_Parameter *param) {
  Node y = primitiv::operators::parameter<Node>(*to_cc(param));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_parameter(primitiv_Parameter *param) {
  Tensor y = primitiv::operators::parameter<Tensor>(*to_cc(param));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_op_tanh(const primitiv_Node *x) {
  Node y = primitiv::operators::tanh<Node>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_tanh(const primitiv_Tensor *x) {
  Tensor y = primitiv::operators::tanh<Tensor>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_op_matmul(const primitiv_Node *a, const primitiv_Node *b) {
  Node y = primitiv::operators::matmul<Node>(*to_cc(a), *to_cc(b));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_matmul(const primitiv_Tensor *a, const primitiv_Tensor *b) {
  Tensor y = primitiv::operators::matmul<Tensor>(*to_cc(a), *to_cc(b));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_op_batch_mean(const primitiv_Node *x) {
  Node y = primitiv::operators::batch::mean<Node>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_op_batch_mean(const primitiv_Tensor *x) {
  Tensor y = primitiv::operators::batch::mean<Tensor>(*to_cc(x));
  return to_c_from_value(y);
}

}  // end extern "C"
