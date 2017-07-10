#include <config.h>

#include <algorithm>
#include <cmath>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/trainer_impl.h>

namespace primitiv {
namespace trainers {

void SGD::configure_parameter(Parameter &param) {}

void SGD::update_parameter(float scale, Parameter &param) {
  param.value() -= (scale * eta_) * param.gradient();
}

void SGD::update_epoch() {}

void Adam::configure_parameter(Parameter &param) {
  for (const char *name : {"adam-m1", "adam-m2"}) {
    if (!param.has_stats(name)) {
      param.add_stats(name, param.shape());
      param.stats(name).reset(0);
    }
  }
}

void Adam::update_parameter(float scale, Parameter &param) {
  const Tensor &g = param.gradient();
  Tensor &m1 = param.stats("adam-m1");
  Tensor &m2 = param.stats("adam-m2");
  m1 = beta1_ * m1 + (1 - beta1_) * g;
  m2 = beta2_ * m2 + (1 - beta2_) * g * g;
  const Tensor mm1 = m1 / (1 - std::pow(beta1_, epoch_));
  const Tensor mm2 = m2 / (1 - std::pow(beta2_, epoch_));
  param.value() -= (scale * alpha_) * mm1 / (tensor_ops::sqrt(mm2) + eps_);
}

void Adam::update_epoch() {
  ++epoch_;
}

}  // namespace trainers
}  // namespace primitiv
