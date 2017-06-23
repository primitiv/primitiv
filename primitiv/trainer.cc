#include <config.h>

#include <primitiv/error.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer.h>

namespace primitiv {

void Trainer::add_parameter(Parameter *param) {
  if (params_.find(param->name()) != params_.end()) {
    THROW_ERROR("Parameter '" << param->name() << "' already registered.");
  }
  params_.insert(std::make_pair(param->name(), param));
}

}  // namespace primitiv
