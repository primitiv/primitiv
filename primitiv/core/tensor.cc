#include <primitiv/config.h>

#include <primitiv/core/device.h>
#include <primitiv/core/shape_ops.h>
#include <primitiv/core/tensor.h>

namespace primitiv {

float Tensor::to_float() const {
  check_valid();
  if (shape_.size() != 1) {
    PRIMITIV_THROW_ERROR(
        "Tensor has more than 1 values. shape = " << shape_.to_string());
  }
  return device_->tensor_to_vector(*this)[0];
}

std::vector<float> Tensor::to_vector() const {
  check_valid();
  return device_->tensor_to_vector(*this);
}

std::vector<std::uint32_t> Tensor::argmax(std::uint32_t dim) const {
  check_valid();
  return device_->argmax(*this, dim);
}

std::vector<std::uint32_t> Tensor::argmin(std::uint32_t dim) const {
  check_valid();
  return device_->argmin(*this, dim);
}

void *Tensor::mutable_handle() {
  check_valid();
  // If the internal memory is shared with other objects, the memory will be
  // duplicated to maintain the safety of other objects.
  if (handle_.use_count() > 1) {
    *this = device_->copy_tensor(*this);
  }
  return handle_.get();
}

void Tensor::reset(float k) {
  check_valid();
  device_->reset_tensor(k, *this);
}

void Tensor::reset_by_array(const float *values) {
  check_valid();
  device_->reset_tensor_by_array(values, *this);
}

void Tensor::reset_by_vector(const std::vector<float> &values) {
  check_valid();
  device_->reset_tensor_by_vector(values, *this);
}

Tensor Tensor::reshape(const Shape &new_shape) const {
  check_valid();
  return Tensor(shape_ops::reshape(shape_, new_shape), *device_, handle_);
}

Tensor Tensor::flatten() const {
  check_valid();
  return Tensor(shape_ops::flatten(shape_), *device_, handle_);
}

Tensor &Tensor::inplace_multiply_const(float k) {
  check_valid();
  device_->inplace_multiply_const(k, *this);
  return *this;
}

Tensor &Tensor::inplace_add(const Tensor &x) {
  check_valid();
  device_->inplace_add(x, *this);
  return *this;
}

Tensor &Tensor::inplace_subtract(const Tensor &x) {
  check_valid();
  device_->inplace_subtract(x, *this);
  return *this;
}

}  // namepsace primitiv
