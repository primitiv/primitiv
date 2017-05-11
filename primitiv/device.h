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
   * Add constant to each element of the Tensor.
   * @param x A Tensor.
   * @param k Constant to add.
   * @return `x + k * ones(x.shape())`
   */
  virtual Tensor add_const(const Tensor &x, const float k) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
