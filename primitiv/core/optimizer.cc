#include <primitiv/config.h>

#include <cmath>
#include <fstream>
#include <primitiv/core/error.h>
#include <primitiv/core/file_format.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/model.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/optimizer.h>
#include <primitiv/msgpack/reader.h>
#include <primitiv/msgpack/writer.h>

namespace primitiv {

void Optimizer::load(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    PRIMITIV_THROW_ERROR("Could not open file: " << path);
  }
  msgpack::Reader reader(ifs);

  std::uint32_t major, minor;
  reader >> major >> minor;
  FileFormat::assert_version(major, minor);

  std::uint32_t datatype;
  reader >> datatype;
  FileFormat::assert_datatype(FileFormat::DataType::OPTIMIZER, datatype);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  reader >> uint_configs >> float_configs;
  set_configs(uint_configs, float_configs);
}

void Optimizer::save(const std::string &path) const {
  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  get_configs(uint_configs, float_configs);

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    PRIMITIV_THROW_ERROR("Could not open file: " << path);
  }
  msgpack::Writer writer(ofs);

  writer << FileFormat::CurrentVersion::MAJOR;
  writer << FileFormat::CurrentVersion::MINOR;
  writer << static_cast<std::uint32_t>(FileFormat::DataType::OPTIMIZER);
  writer << uint_configs << float_configs;
}

void Optimizer::add_inner(Parameter &param) {
  if (params_.find(&param) != params_.end()) {
    // Explicitly allows to call this function multiple time using the same
    // Parameter object.
    return;
  }
  params_.insert(&param);
  configure_parameter(param);
}

void Optimizer::add_inner(const Model &model) {
  for (const auto &kv : model.get_trainable_parameters()) {
    add_inner(*kv.second);
  }
}

void Optimizer::reset_gradients() {
  for (Parameter *param : params_) {
    param->reset_gradient();
  }
}

void Optimizer::update() {
  if (l2_strength_ > 0) {
    // Weight decay
    for (Parameter *param : params_) {
      param->gradient() += l2_strength_ * param->value();
    }
  }

  if (clip_threshold_ > 0) {
    // Gradient clipping
    float sq_norm = 0;
    for (const Parameter *param : params_) {
      const Tensor &g = param->gradient();
      sq_norm += functions::sum(functions::flatten(g * g), 0).to_float();
    }
    if (sq_norm > clip_threshold_ * clip_threshold_) {
      float clip_scale = clip_threshold_ / std::sqrt(sq_norm);
      for (Parameter *param : params_) {
        param->gradient() *= clip_scale;
      }
    }
  }

  for (Parameter *param : params_) {
    update_parameter(lr_scale_, *param);
  }

  ++epoch_;
}

void Optimizer::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  uint_configs.insert(std::make_pair("Optimizer.epoch", epoch_));
  float_configs.insert(std::make_pair("Optimizer.lr_scale", lr_scale_));
  float_configs.insert(std::make_pair("Optimizer.l2_strength", l2_strength_));
  float_configs.insert(std::make_pair("Optimizer.clip_threshold", clip_threshold_));
}

void Optimizer::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
#define SET_CONFIG(dest, cfg, key) { \
  const auto it = cfg.find(key); \
  if (it != cfg.end()) { \
    dest = it->second; \
  } \
}
  SET_CONFIG(epoch_, uint_configs, "Optimizer.epoch");
  SET_CONFIG(lr_scale_, float_configs, "Optimizer.lr_scale");
  SET_CONFIG(l2_strength_, float_configs, "Optimizer.l2_strength");
  SET_CONFIG(clip_threshold_, float_configs, "Optimizer.clip_threshold");
#undef SET_CONFIG
}

}  // namespace primitiv
