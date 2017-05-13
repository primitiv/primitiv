#include <config.h>

#include <sstream>
#include <utility>
#include <primitiv/device.h>
#include <primitiv/tensor.h>

using std::move;

namespace primitiv {

Tensor::Tensor(Tensor &&src)
: shape_(move(src.shape_))
, device_(src.device_)
, data_(src.data_) {
  src.data_ = nullptr;
}

Tensor::~Tensor() {
  if (data_) {
    device_->free(data_);
  }
}

Tensor &Tensor::operator=(Tensor &&src) {
  if (this != &src) {
    device_->free(data_);
    shape_ = move(src.shape_);
    device_ = src.device_;
    data_ = src.data_;
    src.data_ = nullptr;
  }
  return *this;
}

Tensor::Tensor(const Shape &shape, Device *device)
: shape_(shape)
, device_(device) {
  data_ = device_->allocate(sizeof(float) * shape_.size());
}

Tensor::Tensor(
    const Shape &shape,
    Device *device,
    const std::vector<float> &data)
: shape_(shape)
, device_(device) {
  const unsigned shape_size = shape_.size();
  if (data.size() != shape_size) {
    std::stringstream ss;
    ss << "Data sizes mismatched."
       << " requied: " << shape_size << " (" << shape_.to_string() << ")"
       << ", actual: " << data.size();
    throw std::runtime_error(ss.str());
  }
  const unsigned mem_size = sizeof(float) * shape_size;
  data_ = device_->allocate(mem_size);
  device_->copy_to_device(data_, &data[0], mem_size);
}

std::vector<float> Tensor::to_vector() const {
  const unsigned shape_size = shape_.size();
  std::vector<float> ret(shape_size);
  device_->copy_to_host(&ret[0], data_, sizeof(float) * shape_size);
  return ret;
}

}  // namepsace primitiv
