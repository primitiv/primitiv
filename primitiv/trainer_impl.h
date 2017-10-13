#ifndef PRIMITIV_TRAINER_IMPL_H_
#define PRIMITIV_TRAINER_IMPL_H_

#include <primitiv/trainer.h>

namespace primitiv {
namespace trainers {

#define DECL_DEFAULTS(name_) \
public: \
  std::string name() const override { return #name_; } \
  void get_configs( \
      std::unordered_map<std::string, unsigned> &uint_configs, \
      std::unordered_map<std::string, float> &float_configs) const override; \
  void set_configs( \
      const std::unordered_map<std::string, unsigned> &uint_configs, \
      const std::unordered_map<std::string, float> &float_configs) override; \
private: \
  void configure_parameter(Parameter &param) override; \
  void update_parameter(float scale, Parameter &param) override;

/**
 * Simple stochastic gradient descent.
 */
class SGD : public primitiv::Trainer {
  DECL_DEFAULTS(SGD);

public:
  /**
   * Creates a new SGD object.
   * @param eta Learning rate.
   */
  explicit SGD(const float eta = 0.1) : eta_(eta) {}

  /**
   * Returns the learning rate.
   * @return Learning rate.
   */
  float eta() const { return eta_; }

private:
  float eta_;
};

class AdaGrad : public primitiv::Trainer {
  DECL_DEFAULTS(AdaGrad);

public:
  /**
   * Creates a new AdaGrad object.
   * @param eta Learning rate.
   * @param eps Bias of power.
   */
  AdaGrad(float eta = 0.001, float eps = 1e-8)
    : eta_(eta), eps_(eps){}

  /**
   * Returns the hyperparameter eta.
   * @return The value of eta.
   */
	float eta() const {return eta_;}

  /**
   * Returns the hyperparameter eps.
   * @return The value of eps.
   */
	float eps() const {return eps;}

private:
	float eta_;
	float eps_;
}

/**
 * Adam optimizer.
 * https://arxiv.org/abs/1412.6980
 */
class Adam : public primitiv::Trainer {
  DECL_DEFAULTS(Adam);

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

#undef DECL_DEFAULTS

}  // namespace trainers
}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_IMPL_H_
