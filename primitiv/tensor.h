#ifndef PRIMITIV_TENSOR_H_
#define PRIMITIV_TENSOR_H_

#include <vector>
#include <primitiv/shape.h>

namespace primitiv {

class Device;

/**
 * Value with any dimensions.
 */
class Tensor {
public:
  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&);
  Tensor &operator=(const Tensor &) = delete;
  Tensor &operator=(Tensor &&);
  ~Tensor();

  /**
   * Creates an invalid Tensor.
   */
  inline Tensor() : shape_(), device_(nullptr), data_(nullptr) {}

  /**
   * Creates a new uninitialized Tensor.
   * @param shape Shape of the new Tensor.
   * @param device Device object to manage the internal memory.
   */
  Tensor(const Shape &shape, Device *device);

  /**
   * Creates a new Tensor with specific values.
   * @param shape Shape of the new Tensor.
   * @param device Device object to manage the internal memory.
   * @param data List of values to be set to each element.
   *             Each values should be arranged by the column-major order, and
   *             the batch size is assumed as the last dimension.
   */
  Tensor(const Shape &shape, Device *device, const std::vector<float> &data);

  /**
   * Returns the shape of the Tensor.
   * @return Shape of the Tensor.
   */
  const Shape &shape() const { return shape_; }

  /**
   * Returns the Device object related to the internal memory.
   * @return Device object.
   */
  Device *device() const { return device_; }

  /**
   * Returns the raw pointer of the internal memory.
   * @return Pointer of the internal memory.
   */
  void *data() { return data_; }

  /**
   * Returns the raw const-pointer of the internal memory.
   * @return Const-pointer of the internal memory.
   */
  const void *data() const { return data_; }

  /**
   * Returns a copied list of internal values.
   * @return List of copied values.
   */
  std::vector<float> to_vector() const;

  /**
   * Check whether the object is valid or not.
   * @return true if the object is valid, false otherwise.
   * @remarks This returns false when the object is created through the default
   *          constructor or the object had been moved.
   */
  bool valid() const { return !!data_; }

private:
  Shape shape_;
  Device *device_;
  void *data_;
};

}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_H_
