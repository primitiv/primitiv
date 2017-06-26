#include <config.h>

#include <cmath>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/tensor.h>

namespace primitiv {
namespace initializers {

void Constant::apply(Tensor &x) const {
  x.reset(k_);
}

void XavierUniform::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix()) {
    THROW_ERROR(
        "XavierUniform initializer can be used to only matrices or vectors.");
  }
  const float scale = std::sqrt(6. / (s[0] + s[1]));
  x = x.device()->random_uniform(s, -scale, scale);
}

}  // namespace initializers
}  // namespace primitiv
