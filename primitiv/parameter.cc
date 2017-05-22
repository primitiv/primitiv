#include <config.h>

#include <stdexcept>
#include <primitiv/device.h>
#include <primitiv/initializer.h>
#include <primitiv/parameter.h>

namespace primitiv {

Parameter::Parameter(const Shape &shape, Device *device)
: shape_(shape)
, device_(device)
, value_(device->new_tensor(shape))
, grad_(device->new_tensor(shape)) {
  if (shape_.batch_size() > 1) {
    throw std::runtime_error(
        "The batch size of the parameter shape should be 1. "
        "Given shape: " + shape.to_string());
  }
}

void Parameter::reset_value(const Initializer &init) {
  init.apply(value_);
}

void Parameter::reset_gradient() {
  grad_.set_values(0);
}

void Parameter::add_value(const Tensor &diff) {
  value_.add_gradient(diff);
}

void Parameter::add_gradient(const Tensor &diff) {
  grad_.add_gradient(diff);
}

}  // namespace primitiv
