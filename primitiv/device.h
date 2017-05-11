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
   * Adds a constant to each element of the tensor.
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
   * @remarks If the batch size of `a` or `b` is 1, the data is broadcasted to
   *          all minibatches in opposite data.
   */
  virtual Tensor add(const Tensor &a, const Tensor &b) = 0;

  /**
   * Subtracts a constant from each element of the tensor.
   * @param x A tensor.
   * @param k Constant to subtract.
   * @return `x - k * ones(x.shape())`
   */
  virtual Tensor subtract(const Tensor &x, const float k) = 0;

  /**
   * Subtracts a tensor from a tensor initialized by a constant.
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
   */
  virtual Tensor subtract(const Tensor &a, const Tensor &b) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
