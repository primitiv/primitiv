#include <primitiv/config.h>

#include <algorithm>
#include <cmath>

#include <primitiv/core/functions.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/optimizer_impl.h>

namespace primitiv {
namespace optimizers {

#define SET_CONFIG(dest, cfg, key) { \
  const auto it = cfg.find(key); \
  if (it != cfg.end()) { \
    dest = it->second; \
  } \
}

void SGD::configure_parameter(Parameter &) {}

void SGD::update_parameter(float scale, Parameter &param) {
  param.value() -= (scale * eta_) * param.gradient();
}

void SGD::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("SGD.eta", eta_));
}

void SGD::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(eta_, float_configs, "SGD.eta");
}

void MomentumSGD::configure_parameter(Parameter &param) {
  const std::string name = "MomentumSGD.m";
  if (!param.has_stats(name)) {
    param.add_stats(name, param.shape());
  }
}

void MomentumSGD::update_parameter(float scale, Parameter &param) {
  Tensor &m = param.stats("MomentumSGD.m");
  m *= momentum_;
  m -= (scale * eta_) * param.gradient();
  param.value() += m;
}

void MomentumSGD::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("MomentumSGD.eta", eta_));
  float_configs.insert(std::make_pair("MomentumSGD.momentum", momentum_));
}

void MomentumSGD::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(eta_, float_configs, "MomentumSGD.eta");
  SET_CONFIG(momentum_, float_configs, "MomentumSGD.momentum");
}

void AdaGrad::configure_parameter(Parameter &param) {
  const std::string name = "AdaGrad.m";
  if (!param.has_stats(name)) {
    param.add_stats(name, param.shape());
  }
}

void AdaGrad::update_parameter(float scale, Parameter &param) {
  const Tensor &g = param.gradient();
  Tensor &m = param.stats("AdaGrad.m");
  m += g * g;
  param.value() -= (scale * eta_) * g / (functions::sqrt(m) + eps_);
}

void AdaGrad::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("AdaGrad.eta", eta_));
  float_configs.insert(std::make_pair("AdaGrad.eps", eps_));
}

void AdaGrad::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(eta_, float_configs, "AdaGrad.eta");
  SET_CONFIG(eps_, float_configs, "AdaGrad.eps");
}

void RMSProp::configure_parameter(Parameter &param) {
  const std::string name = "RMSProp.m";
  if (!param.has_stats(name)) {
    param.add_stats(name, param.shape());
  }
}

void RMSProp::update_parameter(float scale, Parameter &param) {
  const Tensor &g = param.gradient();
  Tensor &m = param.stats("RMSProp.m");
  m = alpha_ * m + (1 - alpha_) * g * g;
  param.value() -= (scale * eta_) * g / (functions::sqrt(m) + eps_);
}

void RMSProp::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("RMSProp.eta", eta_));
  float_configs.insert(std::make_pair("RMSProp.alpha", alpha_));
  float_configs.insert(std::make_pair("RMSProp.eps", eps_));
}

void RMSProp::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(eta_, float_configs, "RMSProp.eta");
  SET_CONFIG(alpha_, float_configs, "RMSProp.alpha");
  SET_CONFIG(eps_, float_configs, "RMSProp.eps");
}

void AdaDelta::configure_parameter(Parameter &param) {
  for (const char *name : {"AdaDelta.m1", "AdaDelta.m2"}) {
    if (!param.has_stats(name)) {
      param.add_stats(name, param.shape());
    }
  }
}

void AdaDelta::update_parameter(float scale, Parameter &param) {
  const Tensor &g = param.gradient();
  Tensor &m1 = param.stats("AdaDelta.m1");
  Tensor &m2 = param.stats("AdaDelta.m2");
  m2 *= rho_;
  m2 += (1 - rho_) * g * g;
  const Tensor dx = functions::sqrt((m1 + eps_) / (m2 + eps_)) * g;
  m1 *= rho_;
  m1 += (1 - rho_) * dx * dx;
  param.value() -= scale * dx;
}

void AdaDelta::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("AdaDelta.rho", rho_));
  float_configs.insert(std::make_pair("AdaDelta.eps", eps_));
}

void AdaDelta::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(rho_, float_configs, "AdaDelta.rho");
  SET_CONFIG(eps_, float_configs, "AdaDelta.eps");
}

void Adam::configure_parameter(Parameter &param) {
  for (const char *name : {"Adam.m1", "Adam.m2"}) {
    if (!param.has_stats(name)) {
      param.add_stats(name, param.shape());
    }
  }
}

void Adam::update_parameter(float scale, Parameter &param) {
  const std::uint32_t epoch = get_epoch() + 1;
  const Tensor &g = param.gradient();
  Tensor &m1 = param.stats("Adam.m1");
  Tensor &m2 = param.stats("Adam.m2");
  m1 = beta1_ * m1 + (1 - beta1_) * g;
  m2 = beta2_ * m2 + (1 - beta2_) * g * g;
  const Tensor mm1 = m1 / (1 - std::pow(beta1_, epoch));
  const Tensor mm2 = m2 / (1 - std::pow(beta2_, epoch));
  param.value() -= (scale * alpha_) * mm1 / (functions::sqrt(mm2) + eps_);
}

void Adam::get_configs(
    std::unordered_map<std::string, std::uint32_t> &uint_configs,
    std::unordered_map<std::string, float> &float_configs) const {
  Optimizer::get_configs(uint_configs, float_configs);
  float_configs.insert(std::make_pair("Adam.alpha", alpha_));
  float_configs.insert(std::make_pair("Adam.beta1", beta1_));
  float_configs.insert(std::make_pair("Adam.beta2", beta2_));
  float_configs.insert(std::make_pair("Adam.eps", eps_));
}

void Adam::set_configs(
    const std::unordered_map<std::string, std::uint32_t> &uint_configs,
    const std::unordered_map<std::string, float> &float_configs) {
  Optimizer::set_configs(uint_configs, float_configs);
  SET_CONFIG(alpha_, float_configs, "Adam.alpha");
  SET_CONFIG(beta1_, float_configs, "Adam.beta1");
  SET_CONFIG(beta2_, float_configs, "Adam.beta2");
  SET_CONFIG(eps_, float_configs, "Adam.eps");
}

#undef SET_CONFIG

}  // namespace optimizers
}  // namespace primitiv
