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
  virtual Tensor new_tensor(const Shape &shape) = 0;

  /**
   * Provides a new Tensor object with same-value elements.
   * @param shape Shape of the tensor.
   * @param k Constant to initialize elements.
   * @return A new Tensor object.
   */
  virtual Tensor new_tensor(const Shape &shape, const float k) = 0;

  /**
   * Provides a new Tensor object with specific values.
   * @param shape Shape of the tensor.
   * @param values List of internal values.
   * @return A new Tensor object.
   */
  virtual Tensor new_tensor(
      const Shape &shape, const std::vector<float> &values) = 0;

  /**
   * Deallocates the memory of the tensor.
   * @param x Tensor object to be deallocated.
   * @remarks This method should not be used directly by users.
   */
  virtual void delete_tensor(Tensor &x) = 0;

  /**
   * Retrieves internal values of the tensor as a vector.
   * @param x A tensor.
   * @return A list of the internal values.
   * @remarks Each resulting values are ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  virtual std::vector<float> tensor_to_vector(const Tensor &x) = 0;

  /**
   * Reset internal values of the tensor using a constant.
   * @param x A tensor to be updated.
   * @param k A value used to initialize each element.
   */
  virtual void reset_tensor(Tensor &x, const float k) = 0;

  /**
   * Reset internal values of the tensor using specific values.
   * @param x A tensor to be updated.
   * @param values List of each element.
   * @remarks `values.size()` should be same as `x.shape().size()`. Each element
   *          is ordered by the column-major order, and the batch size is
   *          assumed as the last dimension of the tensor.
   */
  virtual void reset_tensor(Tensor &x, const std::vector<float> &values) = 0;

  /**
   * Provides a new Tensor object in which all elements are initialized by
   * the uniform distribution with range (lower, upper].
   * @param shape Shape of the tensor.
   * @param lower Lower bound of values.
   * @param upper Upper bound of values.
   * @return A new Tensor object.
   */
  virtual Tensor random_uniform(
      const Shape &shape, const float lower, const float upper) = 0;

  /**
   * Provides a new Tensor object in which all elements are initialized by
   * the normal distribution with specific mean and standard deviation.
   * @param shape Shape of the tensor.
   * @param mean Mean of the normal distribution.
   * @param sd Standard deviation of the normal distribution.
   * @return A new Tensor object.
   */
  virtual Tensor random_normal(
      const Shape &shape, const float mean, const float sd) = 0;

  /**
   * Duplicates the tensor.
   * @param x A tensor.
   * @return Duplicated tensor.
   */
  virtual Tensor duplicate(const Tensor &x) = 0;

  /**
   * Inverts the sign of each elements.
   * @param x A tensor.
   * @return `-x`
   */
  virtual Tensor negate(const Tensor &x) = 0;

  /**
   * Adds a constant to each element in the tensor.
   * @param x A tensor.
   * @param k Constant to add.
   * @return `x + k * ones(x.shape())`
   */
  virtual Tensor add(const Tensor &x, const float k) = 0;

  /**
   * Adds two tensors.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a + b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  virtual Tensor add(const Tensor &a, const Tensor &b) = 0;

  /**
   * Subtracts a constant from each element in a tensor.
   * @param x A tensor.
   * @param k Constant to subtract.
   * @return `x - k * ones(x.shape())`
   */
  virtual Tensor subtract(const Tensor &x, const float k) = 0;

  /**
   * Subtracts a tensor from a constant.
   * @param k Constant to be subtracted.
   * @param x A tensor.
   * @return `k * ones(x.shape()) - x`
   */
  virtual Tensor subtract(const float k, const Tensor &x) = 0;

  /**
   * Subtracts the second tensor from the first tensor.
   * @param a Tensor to be subtracted.
   * @param b Tensor to subtract.
   * @return `a - b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  virtual Tensor subtract(const Tensor &a, const Tensor &b) = 0;

  /**
   * Multiples each element in a tensor by a constant.
   * @param x A tensor.
   * @param k Multiplier.
   * @return `k * x`
   */
  virtual Tensor multiply(const Tensor &x, const float k) = 0;

  /**
   * Element-wise multiplication of two tensors.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a \circ b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   */
  virtual Tensor multiply(const Tensor &a, const Tensor &b) = 0;

  /**
   * Divides each element in a tensor by a constant.
   * @param x A tensor.
   * @param k Divisor.
   * @return `x / k`
   * @remarks This function won't check the zero-division.
   */
  virtual Tensor divide(const Tensor &x, const float k) = 0;

  /**
   * Divides a constant by each element in a tensor.
   * @param k Constant to be divided.
   * @param x A divisor tensor.
   * @return `k * ones(x.shape()) ./ x`
   * @remarks This function won't check the zero-division.
   */
  virtual Tensor divide(const float k, const Tensor &x) = 0;

  /**
   * Divides the first tensor by the second tensor.
   * @param a Dividend tensor.
   * @param b Divisor tensor.
   * @return `a ./ b`
   * @remarks If the batch size of `a` or `b` is 1, the single-batch side is
   *          broadcasted to all minibatches in the opposite side.
   *          This function won't check the zero-division.
   */
  virtual Tensor divide(const Tensor &a, const Tensor &b) = 0;

  /**
   * Calculates the transposed matrix.
   * @param x A tensor.
   * @return `x^T`
   * @remarks Number of dimensions of `x` should be 0, 1 or 2.
   */
  virtual Tensor transpose(const Tensor &x) = 0;

  /**
   * Calculates the matrix product (dot product) of two matrices.
   * @param a A tensor.
   * @param b Other tensor.
   * @return `a . b`
   * @remarks Number of dimensions of `a` and `b` should be 0, 1 or 2, and the
   *          second dimension of `a` and the first dimension of `b` should be
   *          same.
   */
  virtual Tensor dot(const Tensor &a, const Tensor &b) = 0;

  /**
   * Calculates the exp function.
   * @param x A tensor.
   * @return `exp(x)`
   */
  virtual Tensor exp(const Tensor &x) = 0;

  /**
   * Calculates the tanh function.
   * @param x A tensor.
   * @return `tanh(x)`
   */
  virtual Tensor tanh(const Tensor &x) = 0;

  /**
   * Calculates the logistic sigmoid function.
   * @param x A tensor.
   * @return `sigmoid(x)`
   */
  virtual Tensor sigmoid(const Tensor &x) = 0;

  /**
   * Calculates the step function.
   * @param x A tensor.
   * @return `x >= 0 ? 1 : 0`
   */
  virtual Tensor step(const Tensor &x) = 0;

  /**
   * Calculates the rectifier function.
   * @param x A tensor.
   * @return `max(x, 0)`
   */
  virtual Tensor relu(const Tensor &x) = 0;

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
  virtual void add_gradient(Tensor &a, const Tensor &b) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
