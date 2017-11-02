#include <config.h>

#include <primitiv/error.h>
#include <primitiv/model.h>

namespace primitiv {

void Model::add_parameter(const std::string &name, Parameter &param) {
  if (param_set_.find(&param) != param_set_.end()) {
    THROW_ERROR(
        "Parameter " << &param << " is already registered with the model.");
  }
  if (param_kv_.find(name) != param_kv_.end()) {
    THROW_ERROR(
        "Parameter name \"" << name << "\" already exists in the model.");
  }
  param_set_.emplace(&param);
  param_kv_.emplace(name, &param);
}

std::vector<Parameter *> Model::get_trainable_parameters() const {
  return std::vector<Parameter *>(param_set_.begin(), param_set_.end());
}

}  // namespace primitiv
