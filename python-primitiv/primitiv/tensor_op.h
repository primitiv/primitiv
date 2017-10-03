#ifndef PYTHON_PRIMITIV_TENSOR_OP_H_
#define PYTHON_PRIMITIV_TENSOR_OP_H_

#include <primitiv/tensor.h>

inline primitiv::Tensor &tensor_inplace_multiply_const(primitiv::Tensor &tensor, float k) {
    tensor *= k;
    return tensor;
}

inline primitiv::Tensor &tensor_inplace_add(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor += x;
    return tensor;
}

inline primitiv::Tensor &tensor_inplace_subtract(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor -= x;
    return tensor;
}

#endif
