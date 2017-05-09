#include <config.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <utility>
#include <primitiv/tensor.h>

namespace {

void *my_allocate(const unsigned size) {
  void *data = std::malloc(size);
  if (!data) {
    std::stringstream ss;
    ss << "Memory allocation failed. requested: " << size << ".";
    throw std::runtime_error(ss.str());
  }
  return data;
}

void my_free(void *ptr) {
  std::free(ptr);
}

}  // namespace

namespace primitiv {

Tensor::Tensor(Tensor &&src)
: shape_(std::move(src.shape_)), data_(src.data_) {
  src.data_ = nullptr;
}

Tensor::~Tensor() {
  ::my_free(data_);
}

Tensor &Tensor::operator=(Tensor &&src) {
  if (this != &src) {
    ::my_free(data_);
    shape_ = std::move(src.shape_);
    data_ = src.data_;
    src.data_ = nullptr;
  }
  return *this;
}

Tensor::Tensor(const Shape &shape)
: shape_(shape) {
  data_ = ::my_allocate(sizeof(float) * shape_.size());
}

Tensor::Tensor(const Shape &shape, const std::vector<float> &data)
: shape_(shape) {
  const unsigned shape_size = shape_.size();
  if (data.size() != shape_size) {
    std::stringstream ss;
    ss << "Data size mismatched. "
       << "requied: " << shape_size << " (" << shape_.to_string() << "), "
       << "actual: " << data.size() << ".";
    throw std::runtime_error(ss.str());
  }
  const unsigned mem_size = sizeof(float) * shape_size;
  data_ = ::my_allocate(mem_size);
  std::memcpy(data_, &data[0], mem_size);
}

std::vector<float> Tensor::to_vector() const {
  const unsigned shape_size = shape_.size();
  std::vector<float> ret(shape_size);
  std::memcpy(&ret[0], data_, sizeof(float) * shape_size);
  return ret;
}

}  // namepsace primitiv
