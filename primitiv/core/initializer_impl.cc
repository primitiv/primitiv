#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/device.h>
#include <primitiv/core/error.h>
#include <primitiv/core/initializer_impl.h>
#include <primitiv/core/tensor.h>

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
    PRIMITIV_THROW_ERROR(
        "Identity initializer can be used to only square matrices.");
  }
  x = x.device().identity(s[0]);
}

void XavierUniform::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix()) {
    PRIMITIV_THROW_ERROR(
        "XavierUniform initializer can be used to only matrices or vectors.");
  }
  const float bound = scale_ * std::sqrt(6. / (s[0] + s[1]));
  x = x.device().random_uniform(s, -bound, bound);
}

void XavierNormal::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (!s.is_matrix()) {
    PRIMITIV_THROW_ERROR(
        "XavierNormal initializer can be used to only matrices or vectors.");
  }
  const float sd = scale_ * std::sqrt(2. / (s[0] + s[1]));
  x = x.device().random_normal(s, 0, sd);
}

void XavierUniformConv2D::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (s.depth() > 4) {
    PRIMITIV_THROW_ERROR(
        "XavierUniformConv2D initializer can be used to only tensors with "
        "up to 4 dimensions.");
  }
  const std::uint32_t fan_in = s[0] * s[1] * s[2];
  const std::uint32_t fan_out = s[0] * s[1] * s[3];
  const float bound = scale_ * std::sqrt(6. / (fan_in + fan_out));
  x = x.device().random_uniform(s, -bound, bound);
}

void XavierNormalConv2D::apply(Tensor &x) const {
  const Shape s = x.shape();
  if (s.depth() > 4) {
    PRIMITIV_THROW_ERROR(
        "XavierNormalConv2D initializer can be used to only tensors with "
        "up to 4 dimensions.");
  }
  const std::uint32_t fan_in = s[0] * s[1] * s[2];
  const std::uint32_t fan_out = s[0] * s[1] * s[3];
  const float sd = scale_ * std::sqrt(2. / (fan_in + fan_out));
  x = x.device().random_normal(s, 0, sd);
}

}  // namespace initializers
}  // namespace primitiv
