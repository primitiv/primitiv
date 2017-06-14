#include <config.h>

#include <algorithm>
#include <primitiv/error.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/trainer_impl.h>

namespace primitiv {

void SGDTrainer::add_parameter(Parameter *param) {
  if (std::find(params_.begin(), params_.end(), param) != params_.end()) {
    THROW_ERROR("Parameter already registered.");
  }
  params_.emplace_back(param);
}

void SGDTrainer::reset_gradients() {
  for (Parameter *param : params_) {
    param->reset_gradient();
  }
}

void SGDTrainer::update(const float scale) {
  const float factor = -eta_ * scale;
  for (Parameter *param : params_) {
    param->add_value(factor * param->gradient());
  }
}

}  // namespace primitiv
