#ifndef PRIMITIV_TENSOR_H_
#define PRIMITIV_TENSOR_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <primitiv/error.h>
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

  Tensor(Tensor &&src)
    : shape_(std::move(src.shape_))
    , device_(src.device_)
    , data_(std::move(src.data_)) {
      src.device_ = nullptr;
    }

  Tensor &operator=(const Tensor &) = default;

  Tensor &operator=(Tensor &&src) {
    if (&src != this) {
      shape_ = std::move(src.shape_);
      device_ = src.device_;
      data_ = std::move(src.data_);
      src.device_ = nullptr;
    }
    return *this;
  }

  /**
   * Creates an invalid Tensor.
   */
  Tensor() : shape_(), device_(nullptr), data_() {}

  /**
   * Check whether the object is valid or not.
   * @return true if the object is valid, false otherwise.
   * @remarks This returns false when the object is created through the default
   *          constructor or the object had been moved.
   */
  bool valid() const { return !!device_; }

  /**
   * Returns the shape of the Tensor.
   * @return Shape of the Tensor.
   */
  Shape shape() const {
    if (!valid()) THROW_ERROR("Invalid tensor.");
    return shape_;
  }

  /**
   * Returns the Device object related to the internal memory.
   * @return Device object.
   */
  Device &device() const {
    if (!valid()) THROW_ERROR("Invalid tensor.");
    return *device_;
  }

  /**
   * Returns the raw pointer of the internal memory.
   * @return Pointer of the internal memory.
   */
  void *data();

  /**
   * Returns the raw const-pointer of the internal memory.
   * @return Const-pointer of the internal memory.
   */
  const void *data() const {
    if (!valid()) THROW_ERROR("Invalid tensor.");
    return data_.get();
  }

  /**
   * Retrieves one internal value in the tensor.
   * @return An internal float value.
   * @remarks This function can be used only when the tensor is a scalar and
   *          non-minibatched (i.e., shape() == Shape()).
   */
  float to_float() const;

  /**
   * Retrieves internal values in the tensor as a vector.
   * @return A list of the internal values.
   * @remarks Each resulting values a re ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  std::vector<float> to_vector() const;

  /**
   * Retrieves argmax indices along an axis.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the maximum values.
   */
  std::vector<std::uint32_t> argmax(std::uint32_t dim) const;

  /**
   * Retrieves argmin indices along an axis.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the minimum values.
   */
  std::vector<std::uint32_t> argmin(std::uint32_t dim) const;

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

  /**
   * Directly multiplies a constant.
   * @param k A constant to multiply.
   * @return `*this`
   */
  Tensor &operator*=(float k);

  /**
   * Directly adds a value.
   * @param x A tensor to add.
   * @return `*this`
   */
  Tensor &operator+=(const Tensor &x);

  /**
   * Directly subtracts a value.
   * @param x A tensor to subtract.
   * @return `*this`
   */
  Tensor &operator-=(const Tensor &x);

private:
  /**
   * Creates a new uninitialized Tensor.
   * @param shape Shape of the new Tensor.
   * @param device Device object to manage the internal memory.
   * @param data Pointer of the device-specific object.
   */
  template <typename ShapeT, typename SharedPtrT>
  Tensor(ShapeT &&shape, Device &device, SharedPtrT &&data)
    : shape_(std::forward<ShapeT>(shape))
    , device_(&device)
    , data_(std::forward<SharedPtrT>(data)) {}

  Shape shape_;
  Device *device_;
  std::shared_ptr<void> data_;
};

}  // namespace primitiv

#endif  // PRIMITIV_TENSOR_H_
