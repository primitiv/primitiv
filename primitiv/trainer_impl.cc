#include <config.h>

#include <algorithm>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/trainer_impl.h>

namespace primitiv {
namespace trainers {

void SGD::reset_gradients() {
  for (const auto &kv : params()) {
    kv.second->reset_gradient();
  }
}

void SGD::update(const float scale) {
  const float factor = -eta_ * scale;
  for (const auto &kv : params()) {
    kv.second->add_value(factor * kv.second->gradient());
  }
}

}  // namespace trainers
}  // namespace primitiv
