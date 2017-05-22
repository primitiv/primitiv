#include <config.h>

#include <sstream>
#include <stdexcept>
#include <utility>
#include <primitiv/device.h>
#include <primitiv/tensor.h>

using std::move;

namespace primitiv {

Tensor::Tensor(Tensor &&src)
: shape_(move(src.shape_))
, device_(src.device_)
, data_(src.data_) {
  src.device_ = nullptr;
  src.data_ = nullptr;
}

Tensor::~Tensor() {
  if (valid()) {
    device_->delete_tensor(*this);
  }
}

Tensor &Tensor::operator=(Tensor &&src) {
  if (this != &src) {
    if (valid()) {
      device_->delete_tensor(*this);
    }
    shape_ = move(src.shape_);
    device_ = src.device_;
    data_ = src.data_;
    src.device_ = nullptr;
    src.data_ = nullptr;
  }
  return *this;
}

std::vector<float> Tensor::get_values() const {
  return device_->get_values(*this);
}

void Tensor::set_values(const float k) {
  device_->set_values(*this, k);
}

void Tensor::set_values(const std::vector<float> &values) {
  device_->set_values(*this, values);
}

void Tensor::add_gradient(const Tensor &x) {
  device_->add_gradient(*this, x);
}

}  // namepsace primitiv
