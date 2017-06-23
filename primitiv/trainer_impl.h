#ifndef PRIMITIV_TRAINER_IMPL_H_
#define PRIMITIV_TRAINER_IMPL_H_

#include <primitiv/trainer.h>

namespace primitiv {

/**
 * Simple stochastic gradient descent.
 */
class SGDTrainer : public Trainer {
  SGDTrainer() = delete;
  SGDTrainer(const SGDTrainer &) = delete;
  SGDTrainer(SGDTrainer &&) = delete;
  SGDTrainer &operator=(const SGDTrainer &) = delete;
  SGDTrainer &operator=(SGDTrainer &&) = delete;

public:
  /**
   * Creates a new SGDTrainer object.
   * @param eta Learning rate.
   */
  explicit SGDTrainer(const float eta) : eta_(eta) {}

  void reset_gradients() override;
  void update(float scale) override;

private:
  float eta_;
};

}  // namespace primitiv

#endif  // PRIMITIV_TRAINER_IMPL_H_
