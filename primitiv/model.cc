#include <config.h>

#include <primitiv/error.h>
#include <primitiv/model.h>

namespace primitiv {

void Model::add_parameter(const std::string &name, Parameter &param) {
  if (name_set_.find(name) != name_set_.end()) {
    THROW_ERROR(
        "Name '" << name << "' already exists in the model.");
  }
  if (param_set_.find(&param) != param_set_.end()) {
    THROW_ERROR(
        "Parameter '" << &param << "' already exists in the model.");
  }
  name_set_.emplace(name);
  param_set_.emplace(&param);
  param_kv_.emplace(name, &param);
}

void Model::add_submodel(const std::string &name, Model &model) {
  if (&model == this) {
    THROW_ERROR("Can't add self as a submodel.");
  }
  if (model.has_submodel(*this)) {
    THROW_ERROR("Can't add an ancestor model as a submodel.");
  }
  if (name_set_.find(name) != name_set_.end()) {
    THROW_ERROR(
        "Name '" << name << "' already exists in the model.");
  }
  if (submodel_set_.find(&model) != submodel_set_.end()) {
    THROW_ERROR(
        "Model '" << &model << "' already exists in the model.");
  }
  name_set_.emplace(name);
  submodel_set_.emplace(&model);
  submodel_kv_.emplace(name, &model);
}

std::vector<Parameter *> Model::get_trainable_parameters() const {
  std::vector<Parameter *> params(param_set_.begin(), param_set_.end());
  for (const Model *sm : submodel_set_) {
    const auto sm_params = sm->get_trainable_parameters();
    params.insert(params.end(), sm_params.begin(), sm_params.end());
  }
  return params;
}

bool Model::has_submodel(const Model &model) const {
  for (const Model *sm : submodel_set_) {
    if (sm == &model) return true;
    if (sm->has_submodel(model)) return true;
  }
  return false;
}

}  // namespace primitiv
