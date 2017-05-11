#ifndef PRIMITIV_TENSOR_OPS_H_
#define PRIMITIV_TENSOR_OPS_H_

#include <primitiv/device.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Adds constant to each element in the tensor.
 * @param k Constant to add.
 * @param x Tensor.
 * @return `k * ones(x.shape()) + x`
 */
inline Tensor operator+(const float k, const Tensor &x) {
  return x.device()->add(x, k);
}

/**
 * Adds constant to each element in the tensor.
 * @param x Tensor.
 * @param k Constant to add.
 * @return `x + k * ones(x.shape())`
 */
inline Tensor operator+(const Tensor &x, const float k) {
  return x.device()->add(x, k);
}

/**
 * Adds two tensors.
 * @param a A tensor.
 * @param b Other tensor.
 * @return `a + b`
 */
inline Tensor operator+(const Tensor &a, const Tensor &b) {
  return a.device()->add(a, b);
}

}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_OPS_H_
