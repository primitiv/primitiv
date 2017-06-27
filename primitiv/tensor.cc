#include <config.h>

#include <utility>
#include <primitiv/device.h>
#include <primitiv/shape_ops.h>
#include <primitiv/tensor.h>

using std::move;

namespace primitiv {

Tensor &Tensor::operator=(Tensor &&src) {
  if (this != &src) {
    shape_ = move(src.shape_);
    device_ = src.device_;
    data_ = move(src.data_);
    src.device_ = nullptr;
  }
  return *this;
}

std::vector<float> Tensor::to_vector() const {
  return device_->tensor_to_vector(*this);
}

void Tensor::reset(const float k) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->reset_tensor(*this, k);
}

void Tensor::reset_by_array(const float *values) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->reset_tensor_by_array(*this, values);
}

void Tensor::reset_by_vector(const std::vector<float> &values) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->reset_tensor_by_vector(*this, values);
}

void Tensor::add_gradient(const Tensor &x) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->add_gradient(*this, x);
}

void Tensor::add_gradient_offset(
    const Tensor &x, unsigned dim, unsigned offset) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->add_gradient_offset(*this, x, dim, offset);
}

void Tensor::add_gradient_sparse(
    const Tensor &x, unsigned dim, const std::vector<unsigned> &ids) {
  if (data_.use_count() > 1) *this = device_->copy_tensor(*this);
  device_->add_gradient_sparse(*this, x, dim, ids);
}

Tensor Tensor::reshape(const Shape &new_shape) const {
  return Tensor(shape_ops::reshape(shape_, new_shape), device_, data_);
}

Tensor Tensor::flatten() const {
  return Tensor(shape_ops::flatten(shape_), device_, data_);
}

}  // namepsace primitiv
