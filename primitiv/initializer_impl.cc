#include <config.h>

#include <cmath>
#include <stdexcept>
#include <primitiv/device.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/tensor.h>

namespace primitiv {
namespace initializers {

void Constant::apply(Tensor &x) const {
  x.reset(k_);
}

void XavierUniform::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (s.dims().size() > 2) {
    throw std::runtime_error(
        "XavierUniform initializer can be used to only matrices or vectors.");
  }
  const float scale = std::sqrt(6. / (s.dim(0) + s.dim(1)));
  x = x.device()->random_uniform(s, -scale, scale);
}

}  // namespace initializers
}  // namespace primitiv
