#include <config.h>

#include <utility>
#include <primitiv/device.h>
#include <primitiv/shape_ops.h>
#include <primitiv/tensor.h>

using std::move;

namespace primitiv {

void Tensor::unique() {
  if (data_.use_count() > 1) {
    *this = device_->copy_tensor(*this);
  }
}

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

void *Tensor::data() {
  unique();
  return data_.get();
}

void Tensor::reset(const float k) {
  unique();
  device_->reset_tensor(k, *this);
}

void Tensor::reset_by_array(const float *values) {
  unique();
  device_->reset_tensor_by_array(values, *this);
}

void Tensor::reset_by_vector(const std::vector<float> &values) {
  unique();
  device_->reset_tensor_by_vector(values, *this);
}

Tensor Tensor::reshape(const Shape &new_shape) const {
  return Tensor(shape_ops::reshape(shape_, new_shape), device_, data_);
}

Tensor Tensor::flatten() const {
  return Tensor(shape_ops::flatten(shape_), device_, data_);
}

}  // namepsace primitiv
