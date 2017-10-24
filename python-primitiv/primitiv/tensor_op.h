#ifndef PYTHON_PRIMITIV_TENSOR_OP_H_
#define PYTHON_PRIMITIV_TENSOR_OP_H_

#include <primitiv/operators.h>
#include <primitiv/tensor.h>

namespace python_primitiv_tensor {

using namespace primitiv;

inline Tensor op_tensor_pos(const Tensor &x) {
    return +x;
}

inline Tensor op_tensor_neg(const Tensor &x) {
    return -x;
}

inline Tensor op_tensor_add(const Tensor &x, float k) {
    return x + k;
}

inline Tensor op_tensor_add(float k, const Tensor &x) {
    return k + x;
}

inline Tensor op_tensor_add(const Tensor &a, const Tensor &b) {
    return a + b;
}

inline Tensor op_tensor_sub(const Tensor &x, float k) {
    return x - k;
}

inline Tensor op_tensor_sub(float k, const Tensor &x) {
    return k - x;
}

inline Tensor op_tensor_sub(const Tensor &a, const Tensor &b) {
    return a - b;
}

inline Tensor op_tensor_mul(const Tensor &x, float k) {
    return x * k;
}

inline Tensor op_tensor_mul(float k, const Tensor &x) {
    return k * x;
}

inline Tensor op_tensor_mul(const Tensor &a, const Tensor &b) {
    return a * b;
}

inline Tensor op_tensor_div(const Tensor &x, float k) {
    return x / k;
}

inline Tensor op_tensor_div(float k, const Tensor &x) {
    return k / x;
}

inline Tensor op_tensor_div(const Tensor &a, const Tensor &b) {
    return a / b;
}

inline void op_tensor_imul(primitiv::Tensor &tensor, float k) {
    tensor *= k;
}

inline void op_tensor_iadd(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor += x;
}

inline void op_tensor_isub(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor -= x;
}

}  // namespace python_primitiv_tensor

#endif  // PYTHON_PRIMITIV_TENSOR_OP_H_
