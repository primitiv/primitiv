#include <config.h>

#include <algorithm>
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

}  // namespace trainers
}  // namespace primitiv
