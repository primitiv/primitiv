#include <config.h>

#include <algorithm>
#include <stdexcept>
#include <primitiv/sgd_trainer.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>

namespace primitiv {

void SGDTrainer::add_parameter(Parameter *param) {
  if (std::find(params_.begin(), params_.end(), param) != params_.end()) {
    throw std::runtime_error("Parameter already registered.");
  }
  params_.emplace_back(param);
}

void SGDTrainer::reset_gradients() {
  for (Parameter *param : params_) {
    param->reset_gradient();
  }
}

void SGDTrainer::update() {
  for (Parameter *param : params_) {
    param->add_value(-eta_ * param->gradient());
  }
}

}  // namespace primitiv
