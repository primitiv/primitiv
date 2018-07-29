#ifndef PRIMITIV_CORE_OPTIMIZER_IMPL_H_
#define PRIMITIV_CORE_OPTIMIZER_IMPL_H_

#include <primitiv/core/optimizer.h>

namespace primitiv {
namespace optimizers {

#define PRIMITIV_DECL_DEFAULTS \
public: \
  void get_configs( \
      std::unordered_map<std::string, std::uint32_t> &uint_configs, \
      std::unordered_map<std::string, float> &float_configs) const override; \
  void set_configs( \
      const std::unordered_map<std::string, std::uint32_t> &uint_configs, \
      const std::unordered_map<std::string, float> &float_configs) override; \
private: \
  void configure_parameter(Parameter &param) override; \
  void update_parameter(float scale, Parameter &param) override;

/**
 * Simple stochastic gradient descent.
 */
class SGD : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new SGD object.
   * @param eta Learning rate.
   */
  explicit SGD(float eta = 0.1) : eta_(eta) {}

  /**
   * Returns the learning rate.
   * @return Learning rate.
   */
  float eta() const { return eta_; }

private:
  float eta_;
};

/**
 * Stochastic gradient descent with momentum.
 */
class MomentumSGD : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new MomentumSGD object.
   * @param eta Learning rate.
   * @param momentum Decay factor of the momentum.
   */
  MomentumSGD(float eta = 0.01, float momentum = 0.9)
    : eta_(eta), momentum_(momentum) {}

  /**
   * Returns the hyperparameter eta.
   * @return The value of eta.
   */
  float eta() const { return eta_; }

  /**
   * Returns the hyperparameter momentum.
   * @return The value of momentum.
   */
  float momentum() const { return momentum_; }

private:
  float eta_;
  float momentum_;
};

/**
 * AdaGrad optimizer.
 */
class AdaGrad : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new AdaGrad object.
   * @param eta Learning rate.
   * @param eps Bias of power.
   */
  AdaGrad(float eta = 0.001, float eps = 1e-8)
    : eta_(eta), eps_(eps) {}

  /**
   * Returns the hyperparameter eta.
   * @return The value of eta.
   */
  float eta() const { return eta_; }

  /**
   * Returns the hyperparameter eps.
   * @return The value of eps.
   */
  float eps() const { return eps_; }

private:
  float eta_;
  float eps_;
};

/**
 * RMSProp Optimizer.
 */
class RMSProp : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new RMSProp object.
   * @param eta Learning rate.
   * @param alpha Decay factor of moment.
   * @param eps Bias of power.
   */
  RMSProp(float eta = 0.01, float alpha = 0.9, float eps = 1e-8)
    : eta_(eta), alpha_(alpha), eps_(eps) {}

  /**
   * Returns the hyperparameter eta.
   * @return The value of eta.
   */
  float eta() const { return eta_; }

  /**
   * Returns the hyperparameter alpha.
   * @return The value of alpha.
   */
  float alpha() const { return alpha_; }

  /**
   * Returns the hyperparameter eps.
   * @return The value of eps.
   */
  float eps() const { return eps_; }

private:
  float eta_;
  float alpha_;
  float eps_;
};

/**
 * AdaDelta optimizer.
 * https://arxiv.org/abs/1212.5701
 */
class AdaDelta : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new AdaDelta object.
   * @param rho Decay factor of RMS operation.
   * @param eps Bias of RMS values.
   */
  AdaDelta(float rho = 0.95, float eps = 1e-6)
    : rho_(rho), eps_(eps) {}

  /**
   * Returns the hyperparameter rho.
   * @return The value of rho.
   */
  float rho() const { return rho_; }

  /**
   * Returns the hyperparameter eps.
   * @return The value of eps.
   */
  float eps() const { return eps_; }

private:
  float rho_;
  float eps_;
};

/**
 * Adam optimizer.
 * https://arxiv.org/abs/1412.6980
 */
class Adam : public primitiv::Optimizer {
  PRIMITIV_DECL_DEFAULTS;

public:
  /**
   * Creates a new Adam object.
   * @param alpha Learning rate.
   * @param beta1 Decay factor of momentum history.
   * @param beta2 Decay factor of power history.
   * @param eps Bias of power.
   */
  Adam(
      float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999,
      float eps = 1e-8)
    : alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps) {}

  /**
   * Returns the hyperparameter alpha.
   * @return The value of alpha.
   */
  float alpha() const { return alpha_; }

  /**
   * Returns the hyperparameter beta1.
   * @return The value of beta1.
   */
  float beta1() const { return beta1_; }

  /**
   * Returns the hyperparameter beta2.
   * @return The value of beta2.
   */
  float beta2() const { return beta2_; }

  /**
   * Returns the hyperparameter eps.
   * @return The value of eps.
   */
  float eps() const { return eps_; }

private:
  float alpha_;
  float beta1_;
  float beta2_;
  float eps_;
};

#undef PRIMITIV_DECL_DEFAULTS

}  // namespace optimizers
}  // namespace primitiv

#endif  // PRIMITIV_CORE_OPTIMIZER_IMPL_H_
