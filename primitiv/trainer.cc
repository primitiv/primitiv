#include <config.h>

#include <cmath>
#include <primitiv/error.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/trainer.h>

namespace primitiv {

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
      sq_norm += tensor_ops::sum(tensor_ops::flatten(g * g), 0).to_vector()[0];
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
