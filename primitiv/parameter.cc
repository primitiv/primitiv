#include <config.h>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/initializer.h>
#include <primitiv/parameter.h>

namespace primitiv {

void Parameter::check_shape() {
  if (shape_.batch_size() > 1) {
    THROW_ERROR(
        "The batch size of the parameter shape should be 1. Given shape: "
        << shape_.to_string());
  }
}

Parameter::Parameter(const Shape &shape, Device *device)
: shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
}

Parameter::Parameter(
    const Shape &shape, Device *device, const std::vector<float> & value)
: shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
  reset_value(value);
}

Parameter::Parameter(
    const Shape &shape, Device *device, const Initializer &init)
: shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  check_shape();
  reset_value(init);
}

void Parameter::reset_value(const std::vector<float> &value) {
  value_.reset(value);
}

void Parameter::reset_value(const Initializer &init) {
  init.apply(value_);
}

void Parameter::reset_gradient() {
  grad_.reset(0);
}

void Parameter::add_value(const Tensor &diff) {
  value_.add_gradient(diff);
}

void Parameter::add_gradient(const Tensor &diff) {
  grad_.add_gradient(diff);
}

}  // namespace primitiv
