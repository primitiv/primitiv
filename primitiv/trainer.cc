#include <config.h>

#include <primitiv/error.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer.h>

namespace primitiv {

void Trainer::add_parameter(Parameter *param) {
  if (params_.find(param->name()) != params_.end()) {
    THROW_ERROR("Parameter '" << param->name() << "' is already registered.");
  }
  params_.insert(std::make_pair(param->name(), param));
}

void Trainer::reset_gradients() {
  for (const auto &kv : params_) {
    kv.second->reset_gradient();
  }
}

void Trainer::update(float scale) {
  for (const auto &kv : params_) {
    update_parameter(scale, *kv.second);
  }
  update_epoch();
}

}  // namespace primitiv
