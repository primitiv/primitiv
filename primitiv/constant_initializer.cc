#include <config.h>

#include <primitiv/constant_initializer.h>
#include <primitiv/device.h>
#include <primitiv/shape.h>

namespace primitiv {

Tensor ConstantInitializer::generate(const Shape &shape, Device *device) const {
  return device->constant(shape, k_);
}

}  // namespace primitiv
