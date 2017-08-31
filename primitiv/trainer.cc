#include <config.h>

#include <cmath>
#include <fstream>
#include <primitiv/error.h>
#include <primitiv/messages.pb.h>
#include <primitiv/operators.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer.h>
#include <primitiv/trainer_impl.h>

namespace primitiv {

std::shared_ptr<Trainer> Trainer::load(const std::string &path) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  messages::TrainerConfigs msg;

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  if (!msg.ParseFromIstream(&ifs)) {
    THROW_ERROR("Failed to read Trainer message: " << path);
  }

  const std::string name = msg.name();
  std::shared_ptr<Trainer> trainer;
  if (name == "SGD") trainer.reset(new trainers::SGD());
  else if (name == "Adam") trainer.reset(new trainers::Adam());
  else THROW_ERROR("Unknown trainer name: " << name);

  std::unordered_map<std::string, unsigned> uint_configs(
      msg.uint_configs().begin(), msg.uint_configs().end());
  std::unordered_map<std::string, float> float_configs(
      msg.float_configs().begin(), msg.float_configs().end());

  trainer->epoch_ = uint_configs.at("Trainer.epoch");
  trainer->lr_scale_ = float_configs.at("Trainer.lr_scale");
  trainer->l2_strength_ = float_configs.at("Trainer.l2_strength");
  trainer->clip_threshold_ = float_configs.at("Trainer.clip_threshold");

  trainer->set_configs(uint_configs, float_configs);

  return trainer;
}

void Trainer::save(const std::string &path) const {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;

  uint_configs.insert(std::make_pair("Trainer.epoch", epoch_));
  float_configs.insert(std::make_pair("Trainer.lr_scale", lr_scale_));
  float_configs.insert(std::make_pair("Trainer.l2_strength", l2_strength_));
  float_configs.insert(std::make_pair("Trainer.clip_threshold", clip_threshold_));

  get_configs(uint_configs, float_configs);

  messages::TrainerConfigs msg;
  msg.set_name(name());
  msg.mutable_uint_configs()->insert(uint_configs.begin(), uint_configs.end());
  msg.mutable_float_configs()->insert(float_configs.begin(), float_configs.end());

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    THROW_ERROR("Could not open file: " << path);
  }
  if (!msg.SerializeToOstream(&ofs)) {
    THROW_ERROR("Failed to write Trainer message: " << path);
  }
}

void Trainer::add_parameter(Parameter &param) {
  if (params_.find(param.name()) != params_.end()) {
    THROW_ERROR("Parameter '" << param.name() << "' is already registered.");
  }
  params_.insert(std::make_pair(param.name(), &param));
  configure_parameter(param);
}

void Trainer::reset_gradients() {
  for (const auto &kv : params_) {
    kv.second->reset_gradient();
  }
}

void Trainer::update() {
  if (l2_strength_ > 0) {
    // Weight decay
    for (const auto &kv : params_) {
      kv.second->gradient() += l2_strength_ * kv.second->value();
    }
  }

  if (clip_threshold_ > 0) {
    // Gradient clipping
    float sq_norm = 0;
    for (const auto &kv : params_) {
      const Tensor &g = kv.second->gradient();
      sq_norm += operators::sum(operators::flatten(g * g), 0).to_vector()[0];
    }
    if (sq_norm > clip_threshold_ * clip_threshold_) {
      float clip_scale = clip_threshold_ / std::sqrt(sq_norm);
      for (const auto &kv : params_) {
        kv.second->gradient() *= clip_scale;
      }
    }
  }

  for (const auto &kv : params_) {
    update_parameter(lr_scale_, *kv.second);
  }

  ++epoch_;
}

}  // namespace primitiv
