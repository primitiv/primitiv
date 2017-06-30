#ifndef PRIMITIV_TENSOR_H_
#define PRIMITIV_TENSOR_H_

#include <memory>
#include <vector>
#include <primitiv/shape.h>

namespace primitiv {

class Device;

/**
 * Value with any dimensions.
 */
class Tensor {
  friend Device;

public:
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor &operator=(Tensor &&);

  /**
   * Creates an invalid Tensor.
   */
  Tensor() : shape_(), device_(nullptr), data_() {}

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
  void *data();

  /**
   * Returns the raw const-pointer of the internal memory.
   * @return Const-pointer of the internal memory.
   */
  const void *data() const { return data_.get(); }

  /**
   * Retrieves internal values of the tensor as a vector.
   * @return A list of the internal values.
   * @remarks Each resulting values a re ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  std::vector<float> to_vector() const;

  /**
   * Reset internal values using a constant.
   * @param k A value to be used to initialize each element.
   */
  void reset(const float k);

  /**
   * Reset internal values using a vector.
   * @param values Array of values to be used to initialize each element.
   * @remarks Length of `values` should be equal to `shape().size()`. Each
   *          element should be ordered by the column-major order, and the batch
   *          size is assumed as the last dimension.
   */
  void reset_by_array(const float *values);

  /**
   * Reset internal values using a vector.
   * @param values List of values to be used to initialize each element.
   * @remarks `values.size()` should be equal to `shape().size()`. Each element
   *          should be ordered by the column-major order, and the batch size is
   *          assumed as the last dimension.
   */
  void reset_by_vector(const std::vector<float> &values);

  /**
   * Check whether the object is valid or not.
   * @return true if the object is valid, false otherwise.
   * @remarks This returns false when the object is created through the default
   *          constructor or the object had been moved.
   */
  bool valid() const { return static_cast<bool>(data_); }

  /**
   * Returns a tensor which have the same values and different shape.
   * @param new_shape New shape with batch size 1.
   * @return A new tensor.
   */
  Tensor reshape(const Shape &new_shape) const;

  /**
   * Returns a flattened tensor.
   * @return A new tensor.
   */
  Tensor flatten() const;

private:
  /**
   * Creates a new uninitialized Tensor.
   * @param shape Shape of the new Tensor.
   * @param device Device object to manage the internal memory.
   * @param data Pointer of the device-specific object.
   */
  template <typename ShapeT, typename SharedPtrT>
  Tensor(ShapeT &&shape, Device *device, SharedPtrT &&data)
    : shape_(std::forward<ShapeT>(shape))
    , device_(device)
    , data_(std::forward<SharedPtrT>(data)) {}

  /**
   * Duplicate internal data if current data is used in other place.
   */
  void unique();

  Shape shape_;
  Device *device_;
  std::shared_ptr<void> data_;
};

}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_H_
