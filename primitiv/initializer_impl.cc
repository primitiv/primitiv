#include <primitiv/config.h>

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

void Uniform::apply(Tensor &x) const {
  x = x.device().random_uniform(x.shape(), lower_, upper_);
}

void Normal::apply(Tensor &x) const {
  x = x.device().random_normal(x.shape(), mean_, sd_);
}

void Identity::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix() || s[0] != s[1]) {
    THROW_ERROR(
        "Identity initializer can be used to only square matrices.");
  }
  x = x.device().identity(s[0]);
}

void XavierUniform::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix()) {
    THROW_ERROR(
        "XavierUniform initializer can be used to only matrices or vectors.");
  }
  const float bound = scale_ * std::sqrt(6. / (s[0] + s[1]));
  x = x.device().random_uniform(s, -bound, bound);
}

void XavierNormal::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix()) {
    THROW_ERROR(
        "XavierNormal initializer can be used to only matrices or vectors.");
  }
  const float sd = scale_ * std::sqrt(2. / (s[0] + s[1]));
  x = x.device().random_normal(s, 0, sd);
}

}  // namespace initializers
}  // namespace primitiv
