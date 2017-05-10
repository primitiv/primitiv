#ifndef PRIMITIV_DEVICE_H_
#define PRIMITIV_DEVICE_H_

namespace primitiv {

class Tensor;

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
   *          And users should use the same Device object used for allocate().
   *          Otherwise the Device object will abort the program.
   */
  virtual void free(void *ptr) = 0;

  /**
   * Returns the number of allocated memory blocks.
   * @return Number of memory blocks.
   */
  virtual unsigned num_blocks() const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
