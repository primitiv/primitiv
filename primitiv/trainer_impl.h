#ifndef PRIMITIV_TRAINER_IMPL_H_
#define PRIMITIV_TRAINER_IMPL_H_

#include <primitiv/trainer.h>

namespace primitiv {
namespace trainers {

#define DECL_DEFAULTS(name) \
private: \
  name() = delete; \
  name(const name &) = delete; \
  name(name &&) = delete; \
  name &operator=(const name &) = delete; \
  name &operator=(name &&) = delete; \
  void configure_parameter(Parameter &param) override; \
  void update_parameter(float scale, Parameter &param) override; \
  void update_epoch() override; \
public: \
  virtual ~name() = default;

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

#undef DECL_DEFAULTS

}  // namespace trainers
}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_IMPL_H_
