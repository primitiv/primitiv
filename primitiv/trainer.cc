#include <config.h>

#include <cmath>
#include <fstream>
#include <primitiv/error.h>
#include <primitiv/messages.pb.h>
#include <primitiv/model.h>
#include <primitiv/operators.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer.h>

namespace {

void read_proto(
    const std::string &path,
    primitiv::messages::Trainer &msg) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }

  if (!msg.ParseFromIstream(&ifs)) {
    THROW_ERROR("Failed to read Trainer message: " << path);
  }
}

void write_proto(
    const std::string &path,
    const primitiv::messages::Trainer &msg) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }

  if (!msg.SerializeToOstream(&ofs)) {
    THROW_ERROR("Failed to write Trainer message: " << path);
  }
}

}  // namespace

namespace primitiv {

void Trainer::load(const std::string &path) {
  messages::Trainer msg;
  ::read_proto(path, msg);
  std::unordered_map<std::string, unsigned> uint_configs(
      msg.uint_configs().begin(), msg.uint_configs().end());
  std::unordered_map<std::string, float> float_configs(
      msg.float_configs().begin(), msg.float_configs().end());
  set_configs(uint_configs, float_configs);
}

void Trainer::save(const std::string &path) const {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  get_configs(uint_configs, float_configs);

  messages::Trainer msg;
  msg.mutable_uint_configs()->insert(uint_configs.begin(), uint_configs.end());
  msg.mutable_float_configs()->insert(float_configs.begin(), float_configs.end());
  ::write_proto(path, msg);
}

void Trainer::add_parameter(Parameter &param) {
  if (params_.find(&param) != params_.end()) {
    THROW_ERROR("Parameter '" << &param << "' is already registered.");
  }
  params_.insert(&param);
  configure_parameter(param);
}

void Trainer::add_model(const Model &model) {
  for (Parameter *param : model.get_trainable_parameters()) {
    add_parameter(*param);
  }
}

void Trainer::reset_gradients() {
  for (Parameter *param : params_) {
    param->reset_gradient();
  }
}

void Trainer::update() {
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
      sq_norm += operators::sum(operators::flatten(g * g), 0).to_float();
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

void Trainer::get_configs(
    std::unordered_map<std::string, unsigned> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  uint_configs.insert(std::make_pair("Trainer.epoch", epoch_));
  float_configs.insert(std::make_pair("Trainer.lr_scale", lr_scale_));
  float_configs.insert(std::make_pair("Trainer.l2_strength", l2_strength_));
  float_configs.insert(std::make_pair("Trainer.clip_threshold", clip_threshold_));
}

void Trainer::set_configs(
    const std::unordered_map<std::string, unsigned> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
#define SET_CONFIG(dest, cfg, key) { \
  const auto it = cfg.find(key); \
  if (it == cfg.end()) { \
    THROW_ERROR("Key not found in the trainer config: " << key); \
  } \
  dest = it->second; \
}
  SET_CONFIG(epoch_, uint_configs, "Trainer.epoch");
  SET_CONFIG(lr_scale_, float_configs, "Trainer.lr_scale");
  SET_CONFIG(l2_strength_, float_configs, "Trainer.l2_strength");
  SET_CONFIG(clip_threshold_, float_configs, "Trainer.clip_threshold");
#undef SET_CONFIG
}

}  // namespace primitiv
