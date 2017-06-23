#include <config.h>

#include <utility>
#include <primitiv/device.h>
#include <primitiv/tensor.h>

using std::move;

namespace primitiv {

Tensor::Tensor(Tensor &&src)
: shape_(move(src.shape_))
, device_(src.device_)
, data_(move(src.data_)) {
  src.device_ = nullptr;
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

void Tensor::reset(const float k) {
  device_->reset_tensor(*this, k);
}

void Tensor::reset_by_array(const float *values) {
  device_->reset_tensor_by_array(*this, values);
}

void Tensor::reset_by_vector(const std::vector<float> &values) {
  device_->reset_tensor_by_vector(*this, values);
}

void Tensor::add_gradient(const Tensor &x) {
  device_->add_gradient(*this, x);
}

void Tensor::add_gradient_offset(
    const Tensor &x, unsigned dim, unsigned offset) {
  device_->add_gradient_offset(*this, x, dim, offset);
}

void Tensor::add_gradient_sparse(
    const Tensor &x, unsigned dim, const std::vector<unsigned> &ids) {
  device_->add_gradient_sparse(*this, x, dim, ids);
}

}  // namepsace primitiv
