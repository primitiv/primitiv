#ifndef PRIMITIV_DEVICE_H_
#define PRIMITIV_DEVICE_H_

#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Interface of the allocator/calculator on a specific device.
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
   * Allocates a new memory block on the device.
   * @param size size of the memory.
   * @return Pointer-like handle of the new memory.
   */
  virtual void *allocate(const unsigned size) = 0;

  /**
   * Unallocate an existing memory block.
   * @param ptr Handle of an existing memory.
   * @remarks Users should dispose all handles before exitting the program.
   *          And users should use the same Device object used for `allocate()`.
   *          Otherwise the Device object will abort the program.
   */
  virtual void free(void *ptr) = 0;

  /**
   * Copies memories from host to device.
   * @param dest Handle of the device memory.
   * @param src Pointer of the host memory.
   * @param size Size to copy.
   */
  virtual void copy_to_device(
      void *dest, const void *src, const unsigned size) = 0;

  /**
   * Copies memories from device to host.
   * @param dest Pointer of the host memory.
   * @param src Handle of the device memory.
   * @param size Size to copy.
   */
  virtual void copy_to_host(
      void *src, const void *dest, const unsigned size) = 0;

  /**
   * Returns the number of allocated memory blocks.
   * @return Number of memory blocks.
   */
  virtual unsigned num_blocks() const = 0;

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
   * Directly adds a constant to a tensor.
   * @param x A tensor to be updated.
   * @param k A parameter constant.
   */
  virtual void aug_add(Tensor &x, const float k) = 0;

  /**
   * Directly adds the second tensor to the first tensor.
   * @param a A tensor to be udpated.
   * @param b A parameter tensor.
   */
  virtual void aug_add(Tensor &a, const Tensor &b) = 0;

  /**
   * Directly subtracts a constant from a tensor.
   * @param x A tensor to be updated.
   * @param k A parameter constant.
   */
  virtual void aug_subtract(Tensor &x, const float k) = 0;

  /**
   * Directly subtracts the second tensor from the first tensor.
   * @param a A tensor to be udpated.
   * @param b A parameter tensor.
   */
  virtual void aug_subtract(Tensor &a, const Tensor &b) = 0;

  /**
   * Directly multiplies a tensor by a constant.
   * @param x A tensor to be updated.
   * @param k A parameter constant.
   */
  virtual void aug_multiply(Tensor &x, const float k) = 0;

  /**
   * Directly multiplies the first tensor by the second tensor.
   * @param a A tensor to be udpated.
   * @param b A parameter tensor.
   */
  virtual void aug_multiply(Tensor &a, const Tensor &b) = 0;

  /**
   * Directly divides a tensor by a constant.
   * @param x A tensor to be updated.
   * @param k A parameter constant.
   */
  virtual void aug_divide(Tensor &x, const float k) = 0;

  /**
   * Directly divides the first tensor by the second tensor.
   * @param a A tensor to be udpated.
   * @param b A parameter tensor.
   */
  virtual void aug_divide(Tensor &a, const Tensor &b) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
