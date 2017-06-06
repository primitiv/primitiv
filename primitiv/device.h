#ifndef PRIMITIV_DEVICE_H_
#define PRIMITIV_DEVICE_H_

#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Interface of the Tensor provider.
 */
class Device {
  Device(const Device &) = delete;
  Device(Device &&) = delete;
  Device &operator=(const Device &) = delete;
  Device &operator=(Device &&) = delete;

public:
  Device() = default;
  virtual ~Device() = default;

  /**
   * Provides a new Tensor object on the device.
   * @param shape Shape of the tensor.
   * @return A new Tensor object.
   */
  Tensor new_tensor(const Shape &shape);

  /**
   * Provides a new Tensor object with same-value elements.
   * @param shape Shape of the tensor.
   * @param k Constant to initialize elements.
   * @return A new Tensor object.
   */
  Tensor new_tensor(const Shape &shape, float k);

  /**
   * Provides a new Tensor object with specific values.
   * @param shape Shape of the tensor.
   * @param values List of internal values.
   * @return A new Tensor object.
   */
  Tensor new_tensor(const Shape &shape, const std::vector<float> &values);

  /**
   * Deallocates the memory of the tensor.
   * @param x Tensor object to be deallocated.
   * @remarks This method should not be used directly by users.
   */
  void delete_tensor(Tensor &x);

  /**
   * Retrieves internal values of the tensor as a vector.
   * @param x A tensor.
   * @return A list of the internal values.
   * @remarks Each resulting values are ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  std::vector<float> tensor_to_vector(const Tensor &x);

  /**
   * Reset internal values of the tensor using a constant.
   * @param x A tensor to be updated.
   * @param k A value used to initialize each element.
   */
  void reset_tensor(Tensor &x, float k);

  /**
   * Reset internal values of the tensor using specific values.
   * @param x A tensor to be updated.
   * @param values List of each element.
   * @remarks `values.size()` should be same as `x.shape().size()`. Each element
   *          is ordered by the column-major order, and the batch size is
   *          assumed as the last dimension of the tensor.
   */
  void reset_tensor(Tensor &x, const std::vector<float> &values);

  /**
   * Provides a new Tensor object in which all elements are initialized by
   * the Bernoulli distribution.
   * @param shape Shape of the tensor.
   * @param p Probability to generate 1.
   * @return A new Tensor object.
   */
  Tensor random_bernoulli(const Shape &shape, float p);

  /**
   * Provides a new Tensor object in which all elements are initialized by
   * the uniform distribution with range (lower, upper].
   * @param shape Shape of the tensor.
   * @param lower Lower bound of values.
   * @param upper Upper bound of values.
   * @return A new Tensor object.
   */
  Tensor random_uniform(
      const Shape &shape, float lower, float upper);

  /**
   * Provides a new Tensor object in which all elements are initialized by
   * the normal distribution with specific mean and standard deviation.
   * @param shape Shape of the tensor.
   * @param mean Mean of the normal distribution.
   * @param sd Standard deviation of the normal distribution.
   * @return A new Tensor object.
   */
  Tensor random_normal(const Shape &shape, float mean, float sd);

  /**
   * Provides partial tensor.
   * @param x A tensor.
   * @param dim Target dimension.
   * @param lower The lower bound.
   * @param upper The upper bound.
   * @return `x([lower,upper) in dim)`
   */
  Tensor slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper);

  /**
   * Provides concatenated tensor.
   * @param xs A list of tensor.
   * @param dim Dimension to join.
   * @return `[xs[0], xs[1], ..., xs[n] in dim]`
   */
  Tensor concat(const std::vector<const Tensor *> &xs, unsigned dim);

  /**
   * Duplicates the tensor.
   * @param x A tensor.
   * @return Duplicated tensor.
   */
  Tensor duplicate(const Tensor &x);

  /**
   * Inverts the sign of each elements.
   * @param x A tensor.
   * @return `-x`
   */
  Tensor negate(const Tensor &x);

  /**
   * Adds a constant to each element in the tensor.
   * @param x A tensor.
   * @param k Constant to add.
   * @return `x + k * ones(x.shape())`
   */
  Tensor add(const Tensor &x, float k);

  /**
   * Adds two tensors.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a + b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  Tensor add(const Tensor &a, const Tensor &b);

  /**
   * Subtracts a constant from each element in a tensor.
   * @param x A tensor.
   * @param k Constant to subtract.
   * @return `x - k * ones(x.shape())`
   */
  Tensor subtract(const Tensor &x, float k);

  /**
   * Subtracts a tensor from a constant.
   * @param k Constant to be subtracted.
   * @param x A tensor.
   * @return `k * ones(x.shape()) - x`
   */
  Tensor subtract(float k, const Tensor &x);

  /**
   * Subtracts the second tensor from the first tensor.
   * @param a Tensor to be subtracted.
   * @param b Tensor to subtract.
   * @return `a - b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  Tensor subtract(const Tensor &a, const Tensor &b);

  /**
   * Multiples each element in a tensor by a constant.
   * @param x A tensor.
   * @param k Multiplier.
   * @return `k * x`
   */
  Tensor multiply(const Tensor &x, float k);

  /**
   * Element-wise multiplication of two tensors.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a \circ b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  Tensor multiply(const Tensor &a, const Tensor &b);

  /**
   * Divides each element in a tensor by a constant.
   * @param x A tensor.
   * @param k Divisor.
   * @return `x / k`
   * @remarks This function won't check the zero-division.
   */
  Tensor divide(const Tensor &x, float k);

  /**
   * Divides a constant by each element in a tensor.
   * @param k Constant to be divided.
   * @param x A divisor tensor.
   * @return `k * ones(x.shape()) ./ x`
   * @remarks This function won't check the zero-division.
   */
  Tensor divide(float k, const Tensor &x);

  /**
   * Divides the first tensor by the second tensor.
   * @param a Dividend tensor.
   * @param b Divisor tensor.
   * @return `a ./ b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   *          This function won't check the zero-division.
   */
  Tensor divide(const Tensor &a, const Tensor &b);

  /**
   * Calculates the transposed matrix.
   * @param x A tensor.
   * @return `x^T`
   * @remarks Number of dimensions of `x` should be 0, 1 or 2.
   */
  Tensor transpose(const Tensor &x);

  /**
   * Calculates the matrix product (dot product) of two matrices.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a . b`
   * @remarks Number of dimensions of `a` and `b` should be 0, 1 or 2, and the
   *          second dimension of `a` and the first dimension of `b` should be
   *          same.
   */
  Tensor dot(const Tensor &a, const Tensor &b);

  /**
   * Calculates the exp function.
   * @param x A tensor.
   * @return `exp(x)`
   */
  Tensor exp(const Tensor &x);

  /**
   * Calculates the tanh function.
   * @param x A tensor.
   * @return `tanh(x)`
   */
  Tensor tanh(const Tensor &x);

  /**
   * Calculates the logistic sigmoid function.
   * @param x A tensor.
   * @return `sigmoid(x)`
   */
  Tensor sigmoid(const Tensor &x);

  /**
   * Calculates the step function.
   * @param x A tensor.
   * @return `x >= 0 ? 1 : 0`
   */
  Tensor step(const Tensor &x);

  /**
   * Calculates the rectifier function.
   * @param x A tensor.
   * @return `max(x, 0)`
   */
  Tensor relu(const Tensor &x);

  /**
   * Calculates the sum over minibatches.
   * @param x A tensor.
   * @return `sum(x[0], x[1], ..., x[x.batch_size])`
   */
  Tensor batch_sum(const Tensor &x);

  /**
   * Directly adds the second tensor to the first tensor.
   * @param a A tensor to be udpated.
   * @param b A source tensor.
   * @remarks This method keeps the shape of `a`, and the behavior is
   *          conditioned according to the batch size of `a` and `b`:
   *              a == b: a += b
   *              a == 1: a += batch_sum(b)
   *              b == 1: a += batch_broadcast(b)
   *              otherwise: error.
   */
  void add_gradient(Tensor &a, const Tensor &b);

  /**
   * Same as `add_gradient`, but updates only part of elements in the specified
   * range defined by `dim` and `offset`.
   * @param a A tensor to be updated.
   * @param b A source tensor.
   * @param dim Dimension to determine the range of `a`.
   * @param offset Offset of the dimension `dim`.
   */
  void add_gradient_offset(
      Tensor &a, const Tensor &b, unsigned dim, unsigned offset);

private:
  // device-specific implementations.

  virtual Tensor new_tensor_impl(const Shape &shape) = 0;
  virtual void delete_tensor_impl(Tensor &x) = 0;

  virtual std::vector<float> tensor_to_vector_impl(const Tensor &x) = 0;

  virtual void reset_tensor_impl(Tensor &x, float k) = 0;
  virtual void reset_tensor_impl(
      Tensor &x, const std::vector<float> &values) = 0;

  virtual Tensor random_bernoulli_impl(const Shape &shape, float p) = 0;
  virtual Tensor random_uniform_impl(
      const Shape &shape, float lower, float upper) = 0;
  virtual Tensor random_normal_impl(
      const Shape &shape, float mean, float sd) = 0;

  virtual Tensor slice_impl(
      const Tensor &x, unsigned dim, unsigned lower, unsigned upper) = 0;
  virtual Tensor concat_impl(
      const std::vector<const Tensor *> &xs,
      unsigned dim, const Shape &new_shape) = 0;

  virtual Tensor duplicate_impl(const Tensor &x) = 0;
  virtual Tensor negate_impl(const Tensor &x) = 0;

  virtual Tensor add_impl(const Tensor &x, float k) = 0;
  virtual Tensor add_impl(const Tensor &a, const Tensor &b) = 0;
  virtual Tensor subtract_impl(const Tensor &x, float k) = 0;
  virtual Tensor subtract_impl(float k, const Tensor &x)  = 0;
  virtual Tensor subtract_impl(const Tensor &a, const Tensor &b) = 0;
  virtual Tensor multiply_impl(const Tensor &x, float k) = 0;
  virtual Tensor multiply_impl(const Tensor &a, const Tensor &b) = 0;
  virtual Tensor divide_impl(const Tensor &x, float k) = 0;
  virtual Tensor divide_impl(float k, const Tensor &x)  = 0;
  virtual Tensor divide_impl(const Tensor &a, const Tensor &b) = 0;

  virtual Tensor transpose_impl(const Tensor &x) = 0;
  virtual Tensor dot_impl(const Tensor &a, const Tensor &b) = 0;

  virtual Tensor exp_impl(const Tensor &x) = 0;
  virtual Tensor tanh_impl(const Tensor &x) = 0;
  virtual Tensor sigmoid_impl(const Tensor &x) = 0;
  virtual Tensor step_impl(const Tensor &x) = 0;
  virtual Tensor relu_impl(const Tensor &x) = 0;

  virtual Tensor batch_sum_impl(const Tensor &x) = 0;

  virtual void add_gradient_impl(Tensor &a, const Tensor &b) = 0;
  virtual void add_gradient_offset_impl(
      Tensor &a, const Tensor &b, unsigned dim, unsigned offset) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
