#ifndef PRIMITIV_TENSOR_OPS_H_
#define PRIMITIV_TENSOR_OPS_H_

#include <primitiv/device.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Adds a constant to each element in the tensor.
 * @param x Tensor.
 * @param k Constant to add.
 * @return `x + k * ones(x.shape())`
 */
inline Tensor operator+(const Tensor &x, const float k) {
  return x.device()->add(x, k);
}

/**
 * Adds a constant to each element in the tensor.
 * @param k Constant to add.
 * @param x Tensor.
 * @return `k * ones(x.shape()) + x`
 */
inline Tensor operator+(const float k, const Tensor &x) {
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

/**
 * Subtracts a constant from each element in the tensor.
 * @param x Tensor to be subtracted.
 * @param k Constant to subtract.
 * @return `x - k * ones(x.shape())`
 */
inline Tensor operator-(const Tensor &x, const float k) {
  return x.device()->subtract(x, k);
}

/**
 * Subtracts a tensor from a tensor initialized by a constant.
 * @param k Constant to be subtracted.
 * @param x A tensor.
 * @return `k * ones(x.shape()) - x`
 */
inline Tensor operator-(const float k, const Tensor &x) {
  return x.device()->subtract(k, x);
}

/** Subtracts the second tensor from the first tensor.
 * @param a Tensor to be subtracted.
 * @param b Tensor to subtract.
 * @return `a - b`
 */
inline Tensor operator-(const Tensor &a, const Tensor &b) {
  return a.device()->subtract(a, b);
}

}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_OPS_H_
