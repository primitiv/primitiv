/* Copyright 2017 The primitiv Authors. All Rights Reserved. */

#include <primitiv/functions.h>

#include <vector>

#include "primitiv_c/internal.h"
#include "primitiv_c/functions.h"

using primitiv::Node;
using primitiv::Tensor;

extern "C" {

primitiv_Node *primitiv_node_func_positive(const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::positive(*to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_positive(const primitiv_Node *x,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_positive(x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_positive(const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::positive(*to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_positive(const primitiv_Tensor *x,
                                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_positive(x), status, nullptr);
}

primitiv_Node *primitiv_node_func_negative(const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::negative(*to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_negative(const primitiv_Node *x,
                                                primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_negative(x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_negative(const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::negative(*to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_negative(const primitiv_Tensor *x,
                                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_negative(x), status, nullptr);
}

primitiv_Node *primitiv_node_func_add_node_const(const primitiv_Node *x,
                                                 float k) {
  return to_c_from_value(primitiv::functions::add(*to_cc(x), k));
}
primitiv_Node *safe_primitiv_node_func_add_node_const(const primitiv_Node *x,
                                                      float k,
                                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_add_node_const(x, k), status, nullptr);
}
primitiv_Node *primitiv_node_func_add_const_node(float k,
                                                 const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::add(k, *to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_add_const_node(float k,
                                                      const primitiv_Node *x,
                                                      primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_add_const_node(k, x), status, nullptr);
}
primitiv_Node *primitiv_node_func_add_node_node(const primitiv_Node *a,
                                                const primitiv_Node *b) {
  return to_c_from_value(primitiv::functions::add(*to_cc(a), *to_cc(b)));
}
primitiv_Node *safe_primitiv_node_func_add_node_node(const primitiv_Node *a,
                                                     const primitiv_Node *b,
                                                     primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_add_node_node(a, b), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_add_tensor_const(const primitiv_Tensor *x,
                                                       float k) {
  return to_c_from_value(primitiv::functions::add(*to_cc(x), k));
}
primitiv_Tensor *safe_primitiv_tensor_func_add_tensor_const(
    const primitiv_Tensor *x, float k, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_add_tensor_const(x, k), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_add_const_tensor(
    float k, const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::add(k, *to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_add_const_tensor(
    float k, const primitiv_Tensor *x, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_add_const_tensor(k, x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_add_tensor_tensor(
    const primitiv_Tensor *a, const primitiv_Tensor *b) {
  return to_c_from_value(primitiv::functions::add(*to_cc(a), *to_cc(b)));
}
primitiv_Tensor *safe_primitiv_tensor_func_add_tensor_tensor(
    const primitiv_Tensor *a,
    const primitiv_Tensor *b,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_add_tensor_tensor(a, b), status, nullptr);
}

primitiv_Node *primitiv_node_func_subtract_node_const(const primitiv_Node *x,
                                                      float k) {
  return to_c_from_value(primitiv::functions::subtract(*to_cc(x), k));
}
primitiv_Node *safe_primitiv_node_func_subtract_node_const(
    const primitiv_Node *x, float k, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_subtract_node_const(x, k), status, nullptr);
}
primitiv_Node *primitiv_node_func_subtract_const_node(float k,
                                                      const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::subtract(k, *to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_subtract_const_node(
    float k, const primitiv_Node *x, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_subtract_const_node(k, x), status, nullptr);
}
primitiv_Node *primitiv_node_func_subtract_node_node(const primitiv_Node *a,
                                                     const primitiv_Node *b) {
  return to_c_from_value(primitiv::functions::subtract(*to_cc(a), *to_cc(b)));
}
primitiv_Node *safe_primitiv_node_func_subtract_node_node(
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_subtract_node_node(a, b), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_subtract_tensor_const(
    const primitiv_Tensor *x, float k) {
  return to_c_from_value(primitiv::functions::subtract(*to_cc(x), k));
}
primitiv_Tensor *safe_primitiv_tensor_func_subtract_tensor_const(
    const primitiv_Tensor *x, float k, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_subtract_tensor_const(x, k), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_subtract_const_tensor(
    float k, const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::subtract(k, *to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_subtract_const_tensor(
    float k, const primitiv_Tensor *x, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_subtract_const_tensor(k, x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_subtract_tensor_tensor(
    const primitiv_Tensor *a, const primitiv_Tensor *b) {
  return to_c_from_value(primitiv::functions::subtract(*to_cc(a), *to_cc(b)));
}
primitiv_Tensor *safe_primitiv_tensor_func_subtract_tensor_tensor(
    const primitiv_Tensor *a,
    const primitiv_Tensor *b,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_subtract_tensor_tensor(a, b), status, nullptr);
}

primitiv_Node *primitiv_node_func_multiply_node_const(const primitiv_Node *x,
                                                      float k) {
  return to_c_from_value(primitiv::functions::multiply(*to_cc(x), k));
}
primitiv_Node *safe_primitiv_node_func_multiply_node_const(
    const primitiv_Node *x, float k, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_multiply_node_const(x, k), status, nullptr);
}
primitiv_Node *primitiv_node_func_multiply_const_node(float k,
                                                      const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::multiply(k, *to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_multiply_const_node(
    float k, const primitiv_Node *x, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_multiply_const_node(k, x), status, nullptr);
}
primitiv_Node *primitiv_node_func_multiply_node_node(const primitiv_Node *a,
                                                     const primitiv_Node *b) {
  return to_c_from_value(primitiv::functions::multiply(*to_cc(a), *to_cc(b)));
}
primitiv_Node *safe_primitiv_node_func_multiply_node_node(
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_multiply_node_node(a, b), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_multiply_tensor_const(
    const primitiv_Tensor *x, float k) {
  return to_c_from_value(primitiv::functions::multiply(*to_cc(x), k));
}
primitiv_Tensor *safe_primitiv_tensor_func_multiply_tensor_const(
    const primitiv_Tensor *x, float k, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_multiply_tensor_const(x, k), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_multiply_const_tensor(
    float k, const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::multiply(k, *to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_multiply_const_tensor(
    float k, const primitiv_Tensor *x, primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_multiply_const_tensor(k, x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_multiply_tensor_tensor(
    const primitiv_Tensor *a, const primitiv_Tensor *b) {
  return to_c_from_value(primitiv::functions::multiply(*to_cc(a), *to_cc(b)));
}
primitiv_Tensor *safe_primitiv_tensor_func_multiply_tensor_tensor(
    const primitiv_Tensor *a,
    const primitiv_Tensor *b,
    primitiv_Status *status) {
  SAFE_RETURN(
      primitiv_tensor_func_multiply_tensor_tensor(a, b), status, nullptr);
}

primitiv_Node *primitiv_node_func_divide_node_const(const primitiv_Node *x,
                                                    float k) {
  return to_c_from_value(primitiv::functions::divide(*to_cc(x), k));
}
primitiv_Node *safe_primitiv_node_func_divide_node_const(
    const primitiv_Node *x, float k, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_divide_node_const(x, k), status, nullptr);
}
primitiv_Node *primitiv_node_func_divide_const_node(float k,
                                                    const primitiv_Node *x) {
  return to_c_from_value(primitiv::functions::divide(k, *to_cc(x)));
}
primitiv_Node *safe_primitiv_node_func_divide_const_node(
    float k, const primitiv_Node *x, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_divide_const_node(k, x), status, nullptr);
}
primitiv_Node *primitiv_node_func_divide_node_node(const primitiv_Node *a,
                                                   const primitiv_Node *b) {
  return to_c_from_value(primitiv::functions::divide(*to_cc(a), *to_cc(b)));
}
primitiv_Node *safe_primitiv_node_func_divide_node_node(
    const primitiv_Node *a, const primitiv_Node *b, primitiv_Status *status) {
  SAFE_RETURN(primitiv_node_func_divide_node_node(a, b), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_divide_tensor_const(
    const primitiv_Tensor *x, float k) {
  return to_c_from_value(primitiv::functions::divide(*to_cc(x), k));
}
primitiv_Tensor *safe_primitiv_tensor_func_divide_tensor_const(
    const primitiv_Tensor *x, float k, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_divide_tensor_const(x, k), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_divide_const_tensor(
    float k, const primitiv_Tensor *x) {
  return to_c_from_value(primitiv::functions::divide(k, *to_cc(x)));
}
primitiv_Tensor *safe_primitiv_tensor_func_divide_const_tensor(
    float k, const primitiv_Tensor *x, primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_divide_const_tensor(k, x), status, nullptr);
}
primitiv_Tensor *primitiv_tensor_func_divide_tensor_tensor(
    const primitiv_Tensor *a, const primitiv_Tensor *b) {
  return to_c_from_value(primitiv::functions::divide(*to_cc(a), *to_cc(b)));
}
primitiv_Tensor *safe_primitiv_tensor_func_divide_tensor_tensor(
    const primitiv_Tensor *a,
    const primitiv_Tensor *b,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_tensor_func_divide_tensor_tensor(a, b), status, nullptr);
}

primitiv_Node *primitiv_node_func_mean(const primitiv_Node *x, uint32_t dim) {
  Node y = primitiv::functions::mean<Node>(*to_cc(x), dim);
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_mean(const primitiv_Tensor *x,
                                           uint32_t dim) {
  Tensor y = primitiv::functions::mean<Tensor>(*to_cc(x), dim);
  return to_c_from_value(y);
}

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

primitiv_Node *primitiv_node_func_tanh(const primitiv_Node *x) {
  Node y = primitiv::functions::tanh<Node>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_tanh(const primitiv_Tensor *x) {
  Tensor y = primitiv::functions::tanh<Tensor>(*to_cc(x));
  return to_c_from_value(y);
}

primitiv_Node *primitiv_node_func_matmul(const primitiv_Node *a,
                                         const primitiv_Node *b) {
  Node y = primitiv::functions::matmul<Node>(*to_cc(a), *to_cc(b));
  return to_c_from_value(y);
}

primitiv_Tensor *primitiv_tensor_func_matmul(const primitiv_Tensor *a,
                                             const primitiv_Tensor *b) {
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
