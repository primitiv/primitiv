#ifndef PRIMITIV_TENSOR_OPS_H_
#define PRIMITIV_TENSOR_OPS_H_

#include <primitiv/device.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Duplicates the tensor.
 * @param x A tensor.
 * @return A Duplicated tensor.
 */
inline Tensor operator+(const Tensor &x) {
  return x.device()->duplicate(x);
}

/**
 * Inverts the sign of all elements in the tensor.
 * @param x A tensor.
 * @return `-x`
 */
inline Tensor operator-(const Tensor &x) {
  return x.device()->negate(x);
}

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
 * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
 *          broadcasted to all minibatches in the opposite side.
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
 * Subtracts a tensor from a constant.
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
 * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
 *          broadcasted to all minibatches in the opposite side.
 */
inline Tensor operator-(const Tensor &a, const Tensor &b) {
  return a.device()->subtract(a, b);
}

/**
 * Multiples each element in a tensor by a constant.
 * @param x A tensor.
 * @param k Multiplier.
 * @return `k * x`
 */
inline Tensor operator*(const Tensor &x, const float k) {
  return x.device()->multiply(x, k);
}

/**
 * Multiples each element in a tensor by a constant.
 * @param k A constant.
 * @param x A multiplier tensor.
 * @return `k * x`
 */
inline Tensor operator*(const float k, const Tensor &x) {
  return x.device()->multiply(x, k);
}

/**
 * Element-wise multiplication of two tensors.
 * @param a A tensor.
 * @param b Other tensor.
 * @return `a \circ b`
 * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
 *          broadcasted to all minibatches in the opposite side.
 */
inline Tensor operator*(const Tensor &a, const Tensor &b) {
  return a.device()->multiply(a, b);
}

/**
 * Divides each element in a tensor by a constant.
 * @param x A tensor.
 * @param k Divisor.
 * @return `x / k`
 * @remarks This function won't check the zero-division.
 */
inline Tensor operator/(const Tensor &x, const float k) {
  return x.device()->divide(x, k);
}

/**
 * Divides a constant by each element in a tensor.
 * @param k Constant to be divided.
 * @param x A divisor tensor.
 * @return `k * ones(x.shape()) ./ x`
 * @remarks This function won't check the zero-division.
 */
inline Tensor operator/(const float k, const Tensor &x) {
  return x.device()->divide(k, x);
}

/**
 * Divides the first tensor by the second tensor.
 * @param a Dividend tensor.
 * @param b Divisor tensor.
 * @return `a ./ b`
 * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
 *          broadcasted to all minibatches in the opposite side.
 *          This function won't check the zero-division.
 */
inline Tensor operator/(const Tensor &a, const Tensor &b) {
  return a.device()->divide(a, b);
}

namespace tensor_ops {

/**
 * Calculates the transposed matrix.
 * @param x A tensor.
 * @return `x^T`
 * @remarks Number of dimensions of `x` should be 0, 1 or 2.
 */
inline Tensor transpose(const Tensor &x) {
  return x.device()->transpose(x);
}

/**
 * Calculates the matrix product (dot product) of two matrices.
 * @param a A tensor.
 * @param b Other tensor.
 * @return `a . b`
 * @remarks Number of dimensions of `a` and `b` should be 0, 1 or 2, and the
 *          second dimension of `a` and the first dimension of `b` should be
 *          same.
 */
inline Tensor dot(const Tensor &a, const Tensor &b) {
  return a.device()->dot(a, b);
}

/**
 * Calculates the exp function.
 * @param x A tensor.
 * @return `exp(x)`
 */
inline Tensor exp(const Tensor &x) {
  return x.device()->exp(x);
}

/**
 * Calculates the tanh function.
 * @param x A tensor.
 * @return `tanh(x)`
 */
inline Tensor tanh(const Tensor &x) {
  return x.device()->tanh(x);
}

}  // namespace tensor_ops
}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_OPS_H_
