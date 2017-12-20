#include <primitiv/config.h>

#include <fstream>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/file_format.h>
#include <primitiv/model.h>
#include <primitiv/msgpack/reader.h>
#include <primitiv/msgpack/writer.h>
#include <primitiv/parameter.h>
#include <primitiv/string_utils.h>

namespace primitiv {

void Model::load(const std::string &path, bool with_stats, Device *device) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  msgpack::Reader reader(ifs);

  std::uint32_t major, minor;
  reader >> major >> minor;
  FileFormat::assert_version(major, minor);

  std::uint32_t datatype;
  reader >> datatype;
  FileFormat::assert_datatype(FileFormat::DataType::MODEL, datatype);

  std::uint32_t num_params;
  reader >> num_params;

  const auto params = get_all_parameters();
  for (std::uint32_t i = 0; i < num_params; ++i) {
    std::vector<std::string> key;
    reader >> key;
    const auto it = params.find(key);
    if (it == params.end()) {
      THROW_ERROR(
          "Model does not have a parameter with name: '"
          << string_utils::join(key, ".") << "'");
    }
    it->second->load_inner(
        reader, with_stats, Device::get_reference_or_default(device));
  }
}

void Model::save(const std::string &path, bool with_stats) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  msgpack::Writer writer(ofs);

  writer << FileFormat::CurrentVersion::MAJOR;
  writer << FileFormat::CurrentVersion::MINOR;
  writer << static_cast<std::uint32_t>(FileFormat::DataType::MODEL);

  const auto params = get_all_parameters();
  if (params.size() > 0xffffffffull) {
    THROW_ERROR(
        "Could not store more than 2^32 - 1 parameters in one model file.");
  }
  writer << static_cast<std::uint32_t>(params.size());

  for (const auto &kv : params) {
    writer << kv.first;
    kv.second->save_inner(writer, with_stats);
  }
}

void Model::add(const std::string &name, Parameter &param) {
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

void Model::add(const std::string &name, Model &model) {
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

const Model &Model::get_semiterminal(
    const std::vector<std::string> &names) const {
  const Model *cur = this;
  for (auto it = names.begin(), end = names.end() - 1; it != end; ++it) {
    const auto next = cur->submodel_kv_.find(*it);
    if (next == cur->submodel_kv_.end()) {
      THROW_ERROR(
          "Parameter or submodel not found: "
          "'" << string_utils::join(names, ".") << "'");
    }
    cur = next->second;
  }
  return *cur;
}

const Parameter &Model::get_parameter(
    const std::vector<std::string> &names) const {
  const Model &st = get_semiterminal(names);
  const auto it = st.param_kv_.find(names.back());
  if (it == st.param_kv_.end()) {
    THROW_ERROR(
        "Parameter not found: '" << string_utils::join(names, ".") << "'");
  }
  return *it->second;
}

const Model &Model::get_submodel(const std::vector<std::string> &names) const {
  const Model &st = get_semiterminal(names);
  const auto it = st.submodel_kv_.find(names.back());
  if (it == st.submodel_kv_.end()) {
    THROW_ERROR(
        "Submodel not found: '" << string_utils::join(names, ".") << "'");
  }
  return *it->second;
}

std::map<std::vector<std::string>, Parameter *> Model::get_all_parameters(
    ) const {
  std::map<std::vector<std::string>, Parameter *> params;
  for (const auto &kv : param_kv_) {
    params.emplace(std::vector<std::string> { kv.first }, kv.second);
  }
  for (const auto &sm_kv : submodel_kv_) {
    for (const auto &p_kv : sm_kv.second->get_all_parameters()) {
      std::vector<std::string> key { sm_kv.first };
      key.insert(key.end(), p_kv.first.begin(), p_kv.first.end());
      params.emplace(key, p_kv.second);
    }
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
